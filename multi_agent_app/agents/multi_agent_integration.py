"""
Multi-Agent Integration Layer
Integrates orchestrator, documentation, and redshift agents into the main application
"""

import asyncio
import sys
import os
import json

# Add the parent directory to the path to find the 'agents' module
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .orchestrator_agent import OrchestratorAgent, QueryContext, QueryType
from .documentation_agent import DocumentationAgent
from .redshift_agent import RedshiftQueryAgent


class MultiAgentSystem:
    """
    Main integration class that coordinates all agents.
    """

    def __init__(self, rag_data, redshift_engine, schema_info, llm_client, cache_manager=None):
        self.llm_client = llm_client
        self.cache_manager = cache_manager

        # Initialize specialized agents
        self.orchestrator = OrchestratorAgent(llm_client, cache_manager)
        self.documentation_agent = DocumentationAgent(rag_data, llm_client, cache_manager)
        self.redshift_agent = RedshiftQueryAgent(redshift_engine, schema_info, llm_client, cache_manager)

        # Performance tracking
        self.query_stats = {
            "total_queries": 0,
            "documentation_queries": 0,
            "sql_queries": 0,
            "mixed_queries": 0,
            "avg_response_time": 0.0
        }

    async def process_user_query(self, user_query: str, conversation_history: list = None) -> dict:
        """
        Main entry point for processing user queries through the multi-agent system.
        """
        self.query_stats["total_queries"] += 1

        try:
            # Step 1: Analyze and route the query
            query_context = await self.orchestrator.analyze_query(user_query, conversation_history)

            # Step 2: Route to appropriate agent(s)
            agent_response = await self.orchestrator.route_query(
                query_context,
                self.documentation_agent,
                self.redshift_agent
            )

            # Step 3: Update statistics
            self._update_query_stats(query_context, agent_response)

            # Step 4: Format response for the UI
            formatted_response = self._format_response_for_ui(
                query_context, agent_response
            )

            return formatted_response

        except Exception as e:
            return {
                "success": False,
                "response_text": f"An error occurred in the multi-agent system: {str(e)}",
                "context_data": {},
                "sql_query": None,
                "agent_info": {
                    "error": str(e),
                    "routing_failed": True
                }
            }

    def _update_query_stats(self, query_context: QueryContext, agent_response):
        """Update internal statistics for performance monitoring."""
        # Update query type counters
        if query_context.query_type == QueryType.DOCUMENTATION:
            self.query_stats["documentation_queries"] += 1
        elif query_context.query_type == QueryType.SQL_GENERATION:
            self.query_stats["sql_queries"] += 1
        elif query_context.query_type == QueryType.MIXED:
            self.query_stats["mixed_queries"] += 1

        # Update average response time
        if agent_response.execution_time > 0:
            current_avg = self.query_stats["avg_response_time"]
            total_queries = self.query_stats["total_queries"]
            self.query_stats["avg_response_time"] = (
                (current_avg * (total_queries - 1) + agent_response.execution_time) / total_queries
            )

    def _format_response_for_ui(self, query_context: QueryContext, agent_response) -> dict:
        """Format the agent response for the Streamlit UI."""
        # Get SQL query if available
        sql_query = None
        if agent_response.context_data.get("sql_query"):
            sql_query = agent_response.context_data["sql_query"]
        elif "sql" in agent_response.context_data:
            sql_data = agent_response.context_data["sql"]
            if isinstance(sql_data, dict) and "sql_query" in sql_data:
                sql_query = sql_data["sql_query"]

        context_for_ui = self._format_context_for_ui_display(agent_response.context_data)
        routing_explanation = self.orchestrator.get_routing_explanation(query_context)

        return {
            "success": agent_response.success,
            "response_text": agent_response.response_text,
            "context_data": context_for_ui,
            "sql_query": sql_query,
            "agent_info": {
                "query_type": query_context.query_type.value,
                "confidence": agent_response.confidence,
                "execution_time": agent_response.execution_time,
                "agent_used": agent_response.agent_type.value,
                "routing_explanation": routing_explanation,
                "extracted_entities": query_context.extracted_entities,
                "metadata": agent_response.metadata
            }
        }

    def _format_context_for_ui_display(self, context_data: dict) -> str:
        """Format context data for the UI display as a JSON string."""
        if not context_data:
            return ""

        # Handle different context structures
        if "retrieved_docs" in context_data:
            docs = context_data["retrieved_docs"]
            if docs:
                # Extract only metadata for cleaner UI display
                context_to_show = [doc.get("metadata", {}) for doc in docs if "metadata" in doc]
                return json.dumps(context_to_show, indent=2)
        elif "relevant_schema" in context_data:
            schema = context_data["relevant_schema"]
            if schema:
                return json.dumps(schema, indent=2)
        elif "documentation" in context_data and "sql" in context_data:
            formatted = {
                "documentation_findings": context_data.get("documentation", {}),
                "sql_generation": context_data.get("sql", {})
            }
            return json.dumps(formatted, indent=2)

        # Fallback for any other structure
        try:
            return json.dumps(context_data, indent=2)
        except TypeError:
            return str(context_data)

    def get_performance_metrics(self) -> dict:
        """Get performance metrics for the sidebar display."""
        return {
            "Total Queries": self.query_stats["total_queries"],
            "Documentation Queries": self.query_stats["documentation_queries"],
            "SQL Queries": self.query_stats["sql_queries"],
            "Mixed Queries": self.query_stats["mixed_queries"],
            "Avg Response Time": f"{self.query_stats['avg_response_time']:.2f}s"
        }

    def get_agent_status(self) -> dict:
        """Get the status of all agents."""
        return {
            "orchestrator": "Active",
            "documentation_agent": "Active" if self.documentation_agent.rag_data else "No Data",
            "redshift_agent": "Active" if self.redshift_agent.redshift_engine else "No Connection"
        }


# --- Wrapper Functions for Main Application ---

def create_multi_agent_system(rag_data, redshift_engine, schema_info, llm_client, cache_manager=None):
    """Factory function to create the multi-agent system."""
    return MultiAgentSystem(rag_data, redshift_engine, schema_info, llm_client, cache_manager)


async def process_query_with_agents(multi_agent_system, user_query: str, conversation_history: list = None):
    """Process a query using the multi-agent system."""
    return await multi_agent_system.process_user_query(user_query, conversation_history)