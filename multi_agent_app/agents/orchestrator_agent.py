
"""
Multi-Agent Orchestration System for Redshift RAG Application
Orchestrator Agent - Routes queries to appropriate specialized agents
"""

import re
import json
import asyncio
import logging
from typing import Dict, List, Any
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class QueryType(Enum):
    DOCUMENTATION = "documentation"
    SQL_GENERATION = "sql_generation"
    MIXED = "mixed"
    UNCLEAR = "unclear"


class AgentType(Enum):
    ORCHESTRATOR = "orchestrator"
    DOCUMENTATION = "documentation"
    REDSHIFT_QUERY = "redshift_query"


@dataclass
class QueryContext:
    """Context object passed between agents."""
    user_query: str
    query_type: QueryType
    confidence: float
    extracted_entities: Dict[str, Any]
    conversation_history: List[Dict]
    session_metadata: Dict[str, Any]
    timestamp: datetime


@dataclass
class AgentResponse:
    """Standard response format for all agents."""
    agent_type: AgentType
    success: bool
    response_text: str
    context_data: Dict[str, Any]
    confidence: float
    execution_time: float
    metadata: Dict[str, Any]


class OrchestratorAgent:
    """
    Intelligent orchestrator that analyzes queries and routes them to appropriate agents.
    """

    def __init__(self, llm_client, cache_manager=None):
        self.llm_client = llm_client
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)

        # Query classification patterns
        self.sql_patterns = [
            r'\b(select|insert|update|delete|create|alter|drop)\b',
            r'\b(query|sql|database|table|column|join|where|group by|order by)\b',
            r'\b(count|sum|avg|max|min|distinct)\b',
            r'\bgenerate\s+(sql|query)\b',
            r'\bhow\s+to\s+(query|select|join)\b'
        ]
        self.documentation_patterns = [
            r'\b(what\s+is|what\s+does|explain|describe|definition|meaning)\b',
            r'\b(field|column|table)\s+(description|definition|purpose|means?)\b',
            r'\b(documentation|spec|specification|notes|comments)\b',
            r'\bprogrammer\s+notes?\b',
            r'\bdata\s+(type|length|format)\b'
        ]

        # Entity extraction patterns
        self.field_patterns = [
            r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s+(field|column)\b',
            r'\bfield\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bcolumn\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\b([a-zA-Z_][a-zA-Z0-9_]*_id|[a-zA-Z_][a-zA-Z0-9_]*_name|[a-zA-Z_][a-zA-Z0-9_]*_code)\b'
        ]
        self.table_patterns = [
            r'\btable\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bfrom\s+([a-zA-Z_][a-zA-Z0-9_]*)\b',
            r'\bjoin\s+([a-zA-Z_][a-zA-Z0-9_]*)\b'
        ]

    async def analyze_query(self, user_query: str, conversation_history: List = None) -> QueryContext:
        """Intelligent query analysis using LLM + pattern matching."""
        extracted_entities = self._extract_entities(user_query)
        conv_history = conversation_history or []

        classification_prompt = f"""Analyze this user query and classify it into one of these categories:

1. DOCUMENTATION - User wants to understand field meanings, descriptions, specifications, or data definitions
2. SQL_GENERATION - User wants to generate SQL queries, perform data analysis, or query databases
3. MIXED - Query requires both documentation lookup AND SQL generation
4. UNCLEAR - Query intent is ambiguous and needs clarification

Query: "{user_query}"

Recent conversation context:
{self._format_conversation_history(conv_history[-3:])}

Extracted entities: {extracted_entities}

Respond with JSON:
{{
 "query_type": "DOCUMENTATION|SQL_GENERATION|MIXED|UNCLEAR",
 "confidence": 0.0-1.0,
 "reasoning": "Brief explanation of classification",
 "suggested_approach": "How to handle this query",
 "key_entities": ["list", "of", "important", "entities"]
}}"""

        try:
            llm_response = await self._call_llm_async(classification_prompt)
            classification_result = json.loads(llm_response)

            query_type = QueryType(classification_result.get("query_type", "unclear").lower())
            confidence = float(classification_result.get("confidence", 0.5))
            pattern_confidence = self._pattern_based_classification(user_query)
            final_confidence = (confidence + pattern_confidence["confidence"]) / 2

            return QueryContext(
                user_query=user_query,
                query_type=query_type,
                confidence=final_confidence,
                extracted_entities=extracted_entities,
                conversation_history=conv_history,
                session_metadata={
                    "llm_reasoning": classification_result.get("reasoning", ""),
                    "suggested_approach": classification_result.get("suggested_approach", ""),
                    "pattern_signals": pattern_confidence
                },
                timestamp=datetime.now()
            )

        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}. Falling back to pattern matching.")
            pattern_result = self._pattern_based_classification(user_query)
            return QueryContext(
                user_query=user_query,
                query_type=QueryType(pattern_result["type"]),
                confidence=pattern_result["confidence"],
                extracted_entities=extracted_entities,
                conversation_history=conv_history,
                session_metadata={"fallback": True, "pattern_signals": pattern_result},
                timestamp=datetime.now()
            )

    def _pattern_based_classification(self, query: str) -> Dict[str, Any]:
        """Fallback pattern-based classification."""
        query_lower = query.lower()
        sql_score = sum(1 for p in self.sql_patterns if re.search(p, query_lower))
        doc_score = sum(1 for p in self.documentation_patterns if re.search(p, query_lower))

        if sql_score > doc_score:
            return {
                "type": "sql_generation",
                "confidence": min(sql_score / 3, 1.0),
                "sql_signals": sql_score, "doc_signals": doc_score
            }
        elif doc_score > sql_score:
            return {
                "type": "documentation",
                "confidence": min(doc_score / 3, 1.0),
                "sql_signals": sql_score, "doc_signals": doc_score
            }
        elif sql_score > 0:  # Equal and not zero
            return {
                "type": "mixed", "confidence": 0.7,
                "sql_signals": sql_score, "doc_signals": doc_score
            }
        else:
            return {
                "type": "unclear", "confidence": 0.3,
                "sql_signals": sql_score, "doc_signals": doc_score
            }

    def _extract_entities(self, query: str) -> Dict[str, List[str]]:
        """Extract field names, table names, and other entities from query."""
        entities = {"fields": [], "tables": [], "keywords": []}

        for pattern in self.field_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            # Handle tuples from capturing groups
            entities["fields"].extend([m if isinstance(m, str) else m[0] for m in matches])
        for pattern in self.table_patterns:
            entities["tables"].extend(re.findall(pattern, query, re.IGNORECASE))

        business_terms = re.findall(r'\b(user|member|claim|payment|amount|date|id|code|status)\w*\b', query, re.IGNORECASE)
        entities["keywords"].extend(business_terms)

        for key in entities:
            entities[key] = sorted(list(set(term.lower().strip() for term in entities[key] if term.strip())))
        return entities

    async def route_query(
        self, query_context: QueryContext, documentation_agent, redshift_agent
    ) -> AgentResponse:
        """Route query to appropriate agent(s) based on analysis."""
        start_time = datetime.now()
        try:
            if query_context.query_type == QueryType.DOCUMENTATION:
                response = await documentation_agent.process_query(query_context)
            elif query_context.query_type == QueryType.SQL_GENERATION:
                response = await redshift_agent.process_query(query_context)
            elif query_context.query_type == QueryType.MIXED:
                response = await self._handle_mixed_query(query_context, documentation_agent, redshift_agent)
            else:  # UNCLEAR
                response = await self._handle_unclear_query(query_context)

            response.execution_time = (datetime.now() - start_time).total_seconds()
            return response

        except Exception as e:
            self.logger.error(f"Error routing query: {e}")
            return AgentResponse(
                agent_type=AgentType.ORCHESTRATOR,
                success=False,
                response_text=f"I encountered an error processing your request: {str(e)}",
                context_data={},
                confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )

    async def _handle_mixed_query(self, query_context: QueryContext, doc_agent, sql_agent) -> AgentResponse:
        """Handle queries that need both documentation and SQL generation."""
        doc_response = await doc_agent.process_query(query_context)

        if not doc_response.success:
            return await sql_agent.process_query(query_context)

        enhanced_context = QueryContext(
            user_query=query_context.user_query,
            query_type=QueryType.SQL_GENERATION,  # Now focused on SQL
            confidence=query_context.confidence,
            extracted_entities=query_context.extracted_entities,
            conversation_history=query_context.conversation_history,
            session_metadata={
                **query_context.session_metadata,
                "documentation_context": doc_response.context_data,
                "documentation_response": doc_response.response_text
            },
            timestamp=query_context.timestamp
        )
        sql_response = await sql_agent.process_query(enhanced_context)

        combined_response_text = f"""Based on the documentation, here's what I found:
{doc_response.response_text}
---
And here's the SQL query to help with your request:
{sql_response.response_text}"""

        return AgentResponse(
            agent_type=AgentType.ORCHESTRATOR,
            success=True,
            response_text=combined_response_text,
            context_data={"documentation": doc_response.context_data, "sql": sql_response.context_data},
            confidence=min(doc_response.confidence, sql_response.confidence),
            execution_time=0.0,  # Will be set by caller
            metadata={
                "mixed_query": True,
                "doc_agent_success": doc_response.success,
                "sql_agent_success": sql_response.success
            }
        )

    async def _handle_unclear_query(self, query_context: QueryContext) -> AgentResponse:
        """Handle unclear queries by asking for clarification."""
        clarification_prompt = f"""The user query "{query_context.user_query}" is unclear.

Generate a helpful clarification message that:
1. Acknowledges their question.
2. Asks specific clarifying questions.
3. Provides examples of what they might be looking for.

Extracted entities: {query_context.extracted_entities}

Be conversational and helpful."""

        try:
            clarification_text = await self._call_llm_async(clarification_prompt)
            return AgentResponse(
                agent_type=AgentType.ORCHESTRATOR,
                success=True,
                response_text=clarification_text,
                context_data={"needs_clarification": True},
                confidence=0.7,
                execution_time=0.0,
                metadata={"clarification_request": True}
            )
        except Exception:
            fallback_text = """I'm not sure I understand your question completely. Could you help me by being more specific?

For example:
- Are you looking for information about a specific field or table?
- Do you want to generate a SQL query?
- Are you trying to understand what something means?

Feel free to rephrase your question!"""
            return AgentResponse(
                agent_type=AgentType.ORCHESTRATOR,
                success=True,
                response_text=fallback_text,
                context_data={"needs_clarification": True},
                confidence=0.5,
                execution_time=0.0,
                metadata={"fallback_clarification": True}
            )

    def _format_conversation_history(self, history: List[Dict]) -> str:
        """Format conversation history for LLM context."""
        if not history:
            return "No recent conversation."
        formatted = []
        for entry in history[-3:]:  # Last 3 exchanges
            if hasattr(entry, 'question') and hasattr(entry, 'answer'):
                formatted.append(f"User: {entry.question}")
                formatted.append(f"Assistant: {entry.answer[:200]}...")
        return "\n".join(formatted)

    async def _call_llm_async(self, prompt: str) -> str:
        """Async wrapper for synchronous LLM calls."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.llm_client, prompt)

    def get_routing_explanation(self, query_context: QueryContext) -> str:
        """Generate human-readable explanation of routing decision."""
        confidence_str = f"{query_context.confidence:.0%}"
        explanations = {
            QueryType.DOCUMENTATION: f"I detected this is a documentation question (confidence: {confidence_str}). I'll search the documentation.",
            QueryType.SQL_GENERATION: f"I detected this is a SQL generation request (confidence: {confidence_str}). I'll help create a Redshift query.",
            QueryType.MIXED: f"This question seems to need both documentation and SQL (confidence: {confidence_str}). I'll look up the docs first, then build the query.",
            QueryType.UNCLEAR: f"I'm not entirely sure what you're looking for (confidence: {confidence_str}). Let me ask for clarification."
        }
        explanation = explanations.get(query_context.query_type, "Processing your request...")

        entities = query_context.extracted_entities
        if entities.get("fields") or entities.get("tables"):
            entity_parts = []
            if entities.get("fields"):
                entity_parts.append(f"Fields: {', '.join(entities['fields'])}")
            if entities.get("tables"):
                entity_parts.append(f"Tables: {', '.join(entities['tables'])}")
            explanation += f"\n\nI found these entities: {'; '.join(entity_parts)}"
        return explanation
