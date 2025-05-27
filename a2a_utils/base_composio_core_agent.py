import logging
from typing import Any, Dict, List, Optional

from composio_langchain import ComposioToolSet
from langchain_core.messages import AIMessage

from .base_langgraph_core_agent import BaseLangGraphCoreAgent

logger = logging.getLogger(__name__)

class BaseComposioCoreAgent(BaseLangGraphCoreAgent):
    """
    Core LangGraph agent specialized for using Composio tools.
    Inherits from BaseLangGraphCoreAgent and handles Composio tool initialization.
    """

    DEFAULT_SYSTEM_INSTRUCTION = "You are a helpful assistant. Your primary function is to assist the user with their tasks using the available Composio tools."

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        google_api_key: Optional[str] = None,
        composio_api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        apps: Optional[List[str]] = None,
        llm_provider: str = BaseLangGraphCoreAgent.DEFAULT_LLM_PROVIDER,
        *args,
        **kwargs
    ):
        """
        Initializes the BaseComposioCoreAgent.

        Args:
            openai_api_key: API key for OpenAI. Passed to superclass.
            google_api_key: API key for Google. Passed to superclass.
            composio_api_key: Composio API key. If None, it attempts to use an environment variable.
            model_name: Name of the language model to use. Passed to superclass.
            apps: Optional list of Composio app IDs to enable.
                  If None or empty, default Composio behavior (all connected apps) is used.
            llm_provider: The LLM provider to use ("openai" or "google"). Passed to superclass.
        """
        super().__init__(
            llm_provider=llm_provider,
            openai_api_key=openai_api_key,
            google_api_key=google_api_key,
            model_name=model_name,
            *args,
            **kwargs
        )

        self.composio_toolset: Optional[ComposioToolSet] = None
        self.apps_to_enable = apps if apps else [] # Store for potential re-initialization or logging
        self.composio_api_key = composio_api_key # Store for potential re-initialization

        # Initialize tools and graph immediately if apps are provided or default initialization is fine.
        # The actual fetching of tools and graph creation is deferred to _initialize_graph_if_needed,
        # but we can set up the toolset here.
        if ComposioToolSet:
            self.composio_toolset = ComposioToolSet(api_key=self.composio_api_key)
            # Note: self.tools and self.graph are initialized in _initialize_graph_if_needed
        else:
            logger.error("ComposioToolSet is not available. Cannot initialize Composio tools.")
            # self.tools will remain empty, and graph creation might fail or use no tools.

    def _get_system_instruction(self) -> str:
        """Returns the base system instruction for the Composio agent."""
        # This can be overridden by subclasses if a more specific instruction is needed.
        return self.DEFAULT_SYSTEM_INSTRUCTION

    async def _initialize_graph_if_needed(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Initializes the LangGraph graph with Composio tools.
        Overrides the parent method to include Composio tool fetching.
        """
        if self.graph is None: # Only initialize if not already done
            if not self.model:
                raise ValueError("Language model (self.model) must be initialized.")

            if self.composio_toolset:
                logger.info(f"Fetching Composio tools for apps: {self.apps_to_enable if self.apps_to_enable else 'all connected'}")
                try:
                    # Fetch tools. If self.apps_to_enable is empty, it gets all connected tools.
                    self.tools = self.composio_toolset.get_tools(apps=self.apps_to_enable)
                    if not self.tools:
                        logger.warning("No Composio tools were fetched. The agent might not function as expected.")
                    else:
                        logger.info(f"Successfully fetched {len(self.tools)} Composio tools.")
                except Exception as e:
                    logger.error(f"Failed to fetch Composio tools: {e}", exc_info=True)
                    self.tools = [] # Ensure tools list is empty on failure
            else:
                logger.warning("ComposioToolSet not initialized. No Composio tools will be available.")
                self.tools = []
            
            # Call the superclass's graph initialization logic (which uses self.tools)
            await super()._initialize_graph_if_needed(metadata)
            logger.info(f"BaseComposioCoreAgent graph initialized with {len(self.tools)} tools.")

    async def parse_agent_final_output(self, agent_output: AIMessage) -> Dict[str, Any]:
        """
        Parses the final AIMessage from LangGraph into a structured dictionary for A2A.
        For Composio agents, the output is typically in the AIMessage.content.
        """
        content = agent_output.content
        # Further parsing can be done here if the content is structured (e.g., JSON)
        return {
            "status": "completed", 
            "message_content": content if isinstance(content, str) else str(content)
        }

    # Example of how to add more specific methods for Composio, if needed:
    # def add_composio_app_auth(self, app_id: str, auth_detail: Dict[str, Any]):
    #     """Configures authentication for a specific Composio app."""
    #     if not self.composio_toolset or not AppAuth:
    #         logger.error("ComposioToolSet or AppAuth not available. Cannot configure app authentication.")
    #         return
    #     try:
    #         auth = AppAuth(**auth_detail)
    #         self.composio_toolset.app_auth.add(app_id, auth)
    #         logger.info(f"Authentication configured for Composio app: {app_id}")
    #         # May need to re-initialize graph if tools change based on auth
    #         self.graph = None 
    #         logger.info("Graph will be re-initialized on next execution due to auth change.")
    #     except Exception as e:
    #         logger.error(f"Error configuring Composio app auth for {app_id}: {e}", exc_info=True) 