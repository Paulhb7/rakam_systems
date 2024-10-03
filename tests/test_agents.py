import pytest
import pandas as pd
from typing import Any

from rakam_systems.core import Node, NodeMetadata
from rakam_systems.generation.llm import LLM
from rakam_systems.vector_store import VectorStores
from rakam_systems.generation.agents import ClassifyQuery, RAGGeneration, Agent, Action, MultiAgents

# Simple LLM Implementation
class SimpleLLM(LLM):
    def __init__(self, model="gpt-4o", api_key=None):
        self.model = model
        self.api_key = api_key

    def call_llm(self, sys_prompt: str, prompt: str) -> str:
        return f"Generated response for: {prompt}"

    def call_llm_stream(self, sys_prompt: str, prompt: str):
        responses = ["Streamed", " response", " from", " LLM"]
        for response in responses:
            yield response


# Simple Vector Store Implementation
class SimpleVectorStore(VectorStores):
    def __init__(
        self,
        base_index_path="temp_path",
        embedding_model="all-MiniLM-L6-v2",
    ):
        self.stores = {}

    def create_from_nodes(self, store_name: str, nodes: list):
        self.stores[store_name] = nodes

    def search(self, store_name: str, query: str, number: int = 2):
        nodes = self.stores.get(store_name, [])
        return None, nodes[:number]


class ConcreteAgent(Agent):
    def __init__(self, model: str, api_key: str):
        super().__init__(model=model, api_key=api_key)

    def choose_action(self, input: str, state: Any = None) -> Action:
        # Simply return the action based on the input for testing purposes
        return self.actions.get(input)


@pytest.fixture
def simple_llm():
    return SimpleLLM()


@pytest.fixture
def simple_vector_store():
    vector_store = SimpleVectorStore()
    nodes = [
        Node(
            content="This is a mock content 1.",
            metadata=NodeMetadata(
                source_file_uuid="mock_uuid_1",
                position=None,
                custom={"class_name": "mock_class_1"},
            ),
        ),
        Node(
            content="This is a mock content 2.",
            metadata=NodeMetadata(
                source_file_uuid="mock_uuid_2",
                position=None,
                custom={"class_name": "mock_class_2"},
            ),
        ),
    ]
    vector_store.create_from_nodes("query_classification", nodes)
    return vector_store


@pytest.fixture
def trigger_queries():
    return pd.Series(["trigger query 1", "trigger query 2"])


@pytest.fixture
def class_names():
    return pd.Series(["class 1", "class 2"])


@pytest.fixture
def simple_agent(simple_llm):
    return ConcreteAgent(model="gpt-4o", api_key="dummy_api_key")


### Tests for ClassifyQuery


def test_classify_query_initialization(
    trigger_queries, class_names, simple_vector_store, simple_agent
):
    action = ClassifyQuery(
        simple_agent,
        trigger_queries,
        class_names,
        embedding_model="all-MiniLM-L6-v2",
    )

    assert action.agent == simple_agent
    assert action.trigger_queries.equals(trigger_queries)
    assert action.class_names.equals(class_names)
    assert action.embedding_model == "all-MiniLM-L6-v2"
    # assert isinstance(action.vector_store, SimpleVectorStore)


def test_classify_query_execution(
    trigger_queries, class_names, simple_vector_store, simple_agent
):
    action = ClassifyQuery(
        simple_agent,
        trigger_queries,
        class_names,
        embedding_model="all-MiniLM-L6-v2",
    )
    action.vector_store = simple_vector_store

    query = "test query"
    class_name, matched_trigger_query = action.execute(query=query)

    assert class_name == "mock_class_1"
    assert matched_trigger_query == "This is a mock content 1."


### Tests for RAGGeneration


def test_rag_generation_initialization(simple_vector_store, simple_llm, simple_agent):
    action = RAGGeneration(
        simple_agent,
        "System Prompt",
        "Prompt {query}",
        simple_vector_store,
        vs_descriptions={"query_classification": "Mock Description"},
    )

    assert action.agent == simple_agent
    assert action.sys_prompt == "System Prompt"
    assert action.prompt == "Prompt {query}"
    assert action.vector_stores == simple_vector_store
    assert action.vs_descriptions["query_classification"] == "Mock Description"


# def test_rag_generation_non_stream(simple_vector_store, simple_llm, simple_agent):
#     action = RAGGeneration(
#         simple_agent, "System Prompt", "Prompt {query}", simple_vector_store
#     )

#     result = action.execute("test query", stream=False)

#     # assert result == "Generated response for: Prompt test query"


# def test_rag_generation_stream(simple_vector_store, simple_llm, simple_agent):
#     action = RAGGeneration(
#         simple_agent, "System Prompt", "Prompt {query}", simple_vector_store
#     )

#     result = list(action.execute("test query", stream=True))

#     # assert result == ["Streamed", " response", " from", " LLM"]


### Tests for Agent


def test_agent_initialization(simple_llm):
    agent = ConcreteAgent(model="gpt-4o", api_key="dummy_api_key")
    assert agent.llm.model == "gpt-4o"
    assert agent.state == {}
    assert agent.actions == {}


def test_agent_add_action(simple_agent):
    action = ClassifyQuery(simple_agent, pd.Series(["query"]), pd.Series(["class"]))

    simple_agent.add_action("classify_query", action)

    assert "classify_query" in simple_agent.actions
    assert simple_agent.actions["classify_query"] == action


def test_agent_execute_action(simple_agent):
    action = ClassifyQuery(simple_agent, pd.Series(["query"]), pd.Series(["class"]))
    simple_agent.add_action("classify_query", action)

    # Override choose_action to return the correct action
    simple_agent.choose_action = lambda x, state=None: action

    result = simple_agent.execute_action("classify_query", query="test query")

    assert result is not None

from crewai import Agent as CrewAgent, Task as CrewTask, Process
from crewai import Crew  # Import Crew for orchestrating agents
from multiagents_module import MultiAgents  # Assuming your MultiAgents class is in this module


### Tests for MultiAgents

@pytest.fixture
def crew_agent_1():
    return CrewAgent(
        role='Query Classifier',
        goal='Classify incoming queries into categories',
        backstory='An expert in query classification.'
    )


@pytest.fixture
def crew_agent_2():
    return CrewAgent(
        role='Prompt Generator',
        goal='Generate precise prompts based on user queries.',
        backstory='A master prompt engineer with a deep understanding of NLP.'
    )


@pytest.fixture
def crew_task_1(crew_agent_1):
    return CrewTask(
        description='Classify the given query.',
        expected_output='A classified query.',
        agent=crew_agent_1
    )


@pytest.fixture
def crew_task_2(crew_agent_2):
    return CrewTask(
        description='Generate a well-structured prompt for the LLM.',
        expected_output='A prompt for the LLM.',
        agent=crew_agent_2
    )


### Test for MultiAgents class initialization

def test_multi_agents_initialization(crew_agent_1, crew_agent_2, crew_task_1, crew_task_2):
    agents = [crew_agent_1, crew_agent_2]
    tasks = [crew_task_1, crew_task_2]
    
    # Create MultiAgents system
    multi_agents_system = MultiAgents(agents=agents, tasks=tasks, process=Process.sequential)

    # Check that agents and tasks are correctly set
    assert len(multi_agents_system.crew.agents) == 2
    assert len(multi_agents_system.crew.tasks) == 2
    assert isinstance(multi_agents_system.crew.process, Process)


### Test for MultiAgents kickoff execution

def test_multi_agents_kickoff(crew_agent_1, crew_agent_2, crew_task_1, crew_task_2):
    agents = [crew_agent_1, crew_agent_2]
    tasks = [crew_task_1, crew_task_2]

    # Create MultiAgents system
    multi_agents_system = MultiAgents(agents=agents, tasks=tasks, process=Process.sequential)

    # Define input for the system
    inputs = {'query': 'How to integrate AI in healthcare?'}

    # Kickoff the process
    result = multi_agents_system.kickoff(inputs=inputs)

    # Ensure the result is valid (adjust based on expected result format)
    assert result is not None
    assert isinstance(result, dict) or isinstance(result, str)  # Assuming result could be dict or string

