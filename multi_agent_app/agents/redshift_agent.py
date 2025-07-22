import json
import asyncio
import logging
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re
import sqlparse
from sqlalchemy import text

from .orchestrator_agent import AgentResponse, AgentType, QueryContext
from .metadata_manager import SchemaMetadataManager


class EnhancedRedshiftQueryAgent:
    """
    Enhanced Redshift agent with metadata-aware SQL generation and query execution
    """
    
    def __init__(self, redshift_engine, schema_info, llm_client, cache_manager=None):
        self.redshift_engine = redshift_engine
        self.schema_info = schema_info
        self.llm_client = llm_client
        self.cache_manager = cache_manager
        self.logger = logging.getLogger(__name__)
        # Initialize metadata manager
        self.metadata_manager = SchemaMetadataManager()
        # Parse schema for intelligent query generation
        self.parsed_schema = self._parse_schema_info(schema_info) if schema_info else {}
        # Enhanced prompts with metadata awareness
        self.prompt_templates = {
            "simple_select": """You are an expert Redshift SQL analyst with access to detailed schema metadata and descriptions.

Enhanced Schema Information:
{enhanced_schema_info}

User Request: {user_query}

Conversation Context:
{conversation_history}

Documentation Context:
{documentation_context}

Generate a precise Redshift SQL query that:
1. Uses the exact table and column names from the enhanced schema
2. Leverages the provided descriptions to understand data context
3. Includes appropriate WHERE clauses based on business logic
4. Uses proper Redshift syntax and optimizations
5. Includes LIMIT clause for performance
6. Considers table relationships for accurate joins

Return the SQL query in a code block:
```sql
YOUR_QUERY_HERE
```

Provide a clear explanation of:
- What business question the query answers
- Key tables and columns used
- Any assumptions made based on the metadata""",

            "complex_analytics": """You are an expert Redshift analyst specializing in complex analytics with full schema metadata awareness.

Enhanced Schema Information:
{enhanced_schema_info}

User Request: {user_query}

Conversation Context:
{conversation_history}

Documentation Context:
{documentation_context}

Generate a comprehensive analytical Redshift query that:
1. Uses advanced SQL functions (window functions, CTEs, aggregations)
2. Leverages table relationships from metadata for proper joins
3. Applies business logic based on column descriptions
4. Handles data quality considerations (NULLs, duplicates)
5. Optimizes for Redshift's columnar architecture
6. Includes meaningful column aliases based on business context

Return the SQL query in a code block:
```sql
YOUR_QUERY_HERE
```

Explain:
- The analytical approach and business insights
- Key relationships and join strategies used
- Performance optimizations applied
- How metadata influenced query design""",

            "join_query": """You are an expert Redshift analyst with access to comprehensive table relationship metadata.

Enhanced Schema Information:
{enhanced_schema_info}

User Request: {user_query}

Conversation Context:
{conversation_history}

Documentation Context:
{documentation_context}

Generate an optimal JOIN query that:
1. Uses detected table relationships from metadata
2. Applies appropriate JOIN types based on business logic
3. Considers data integrity and referential constraints
4. Optimizes join order for Redshift performance
5. Handles potential data quality issues
6. Uses meaningful table aliases

Return the SQL query in a code block:
```sql
YOUR_QUERY_HERE
```

Explain:
- The joining strategy and relationship logic
- Why specific JOIN types were chosen
- How metadata relationships guided the design
- Performance considerations for large datasets""",

            "aggregation_query": """You are an expert Redshift analyst with detailed understanding of data distributions and business metrics.

Enhanced Schema Information:
{enhanced_schema_info}

User Request: {user_query}

Conversation Context:
{conversation_history}

Documentation Context:
{documentation_context}

Generate a sophisticated aggregation query that:
1. Uses appropriate aggregate functions based on column metadata
2. Groups data logically based on business dimensions
3. Applies HAVING clauses for meaningful filtering
4. Uses window functions for advanced analytics
5. Handles date/time aggregations with business calendar logic
6. Optimizes for Redshift's distribution and sort keys

Return the SQL query in a code block:
```sql
YOUR_QUERY_HERE
```

Explain:
- The business metrics being calculated
- Grouping strategy and dimensional analysis
- How column metadata influenced aggregation choices
- Performance optimizations for large-scale aggregations"""
        }

    def _extract_sql_from_response(self, response: str) -> str:
        """Extract SQL code block from LLM response using regex."""
        match = re.search(r'```sql\s*(.*?)\s*```', response, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else response.strip()

    def _extract_explanation_from_response(self, response: str, sql_query: str) -> str:
        """Extract explanation by removing SQL code block from LLM response."""
        if sql_query in response:
            return response.replace(sql_query, '').replace('```sql', '').replace('```', '').strip()
        return response.strip()

    def _validate_and_optimize_sql(self, sql_query: str) -> str:
        """Format and validate SQL using sqlparse."""
        if not sql_query or sql_query.startswith('--'):
            return sql_query
        try:
            formatted = sqlparse.format(sql_query, reindent=True, keyword_case='upper')
            return formatted
        except Exception:
            return sql_query

    def _serialize_context(self, context: Dict[str, Any]) -> str:
        """Serialize context dict to JSON string."""
        try:
            return json.dumps(context)
        except Exception:
            return str(context)

    def _deserialize_context(self, context_str: str) -> Dict[str, Any]:
        """Deserialize context JSON string to dict."""
        try:
            return json.loads(context_str)
        except Exception:
            return {}

    async def _call_llm_async(self, prompt: str) -> str:
        """Async LLM call using asyncio."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.llm_client, prompt)

    async def process_query(self, query_context: QueryContext) -> AgentResponse:
        start_time = datetime.now()
        try:
            if not self.redshift_engine or not self.schema_info:
                return AgentResponse(
                    agent_type=AgentType.REDSHIFT_QUERY,
                    success=False,
                    response_text="Redshift connection or schema information is not available.",
                    context_data={},
                    confidence=0.0,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"connection_error": True}
                )

            enhanced_schema = self.metadata_manager.get_enhanced_schema_for_query(
                query_context.user_query,
                query_context.extracted_entities
            )

            if enhanced_schema.get('total_tables_found', 0) == 0:
                await self._initialize_metadata_if_needed()
                enhanced_schema = self.metadata_manager.get_enhanced_schema_for_query(
                    query_context.user_query,
                    query_context.extracted_entities
                )

            sql_query_type = self._classify_sql_query(query_context)

            sql_response = await self._generate_enhanced_sql_query(
                query_context, sql_query_type, enhanced_schema
            )

            validated_sql = self._validate_and_optimize_sql(sql_response["sql"])

            execution_result = None
            if query_context.session_metadata.get("execute_query", False):
                execution_result = await self._execute_query_safely(validated_sql)

            confidence = self._calculate_enhanced_confidence(
                query_context, sql_response, enhanced_schema
            )

            return AgentResponse(
                agent_type=AgentType.REDSHIFT_QUERY,
                success=True,
                response_text=sql_response["explanation"],
                context_data={
                    "sql_query": validated_sql,
                    "query_type": sql_query_type,
                    "enhanced_schema": enhanced_schema,
                    "execution_result": execution_result,
                    "metadata_quality": self._assess_metadata_quality(enhanced_schema)
                },
                confidence=confidence,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "sql_generated": True,
                    "query_complexity": sql_query_type,
                    "metadata_enhanced": True,
                    "tables_found": enhanced_schema.get('total_tables_found', 0),
                    "relationships_used": len(enhanced_schema.get('relationships', []))
                }
            )
        except Exception as e:
            return AgentResponse(
                agent_type=AgentType.REDSHIFT_QUERY,
                success=False,
                response_text=f"I encountered an error: {str(e)}",
                context_data={},
                confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )

    async def _initialize_metadata_if_needed(self):
        try:
            result = self.metadata_manager.extract_and_store_schema(
                self.redshift_engine,
                self.llm_client
            )
            if result.get("status") == "success":
                self.logger.info(f"Metadata initialized: {result['tables_processed']} tables processed")
        except Exception as e:
            self.logger.warning(f"Failed to initialize metadata: {e}")

    async def _generate_enhanced_sql_query(self, query_context: QueryContext, sql_query_type: str, enhanced_schema: Dict[str, Any]) -> Dict[str, str]:
        enhanced_schema_text = self.metadata_manager.get_formatted_schema_for_llm(enhanced_schema)
        conversation_history = self._format_conversation_history(query_context.conversation_history)

        documentation_context = ""
        if "documentation_context" in query_context.session_metadata:
            doc_data = query_context.session_metadata["documentation_context"]
            if isinstance(doc_data, dict) and "retrieved_docs" in doc_data:
                documentation_context = self._format_documentation_context(doc_data["retrieved_docs"])

        prompt_template = self.prompt_templates.get(sql_query_type, self.prompt_templates["simple_select"])

        full_prompt = prompt_template.format(
            enhanced_schema_info=enhanced_schema_text,
            user_query=query_context.user_query,
            conversation_history=conversation_history,
            documentation_context=documentation_context
        )

        try:
            response = await self._call_llm_async(full_prompt)
            sql_query = self._extract_sql_from_response(response)
            explanation = self._extract_explanation_from_response(response, sql_query)
            self._update_pattern_usage_if_matched(sql_query, enhanced_schema)
            return {"sql": sql_query, "explanation": explanation}
        except Exception as e:
            return {"sql": "-- Error generating SQL query", "explanation": f"Error: {str(e)}"}

    async def _execute_query_safely(self, sql_query: str) -> Optional[Dict[str, Any]]:
        if not sql_query or sql_query.startswith("--"):
            return None
        try:
            safe_sql = self._add_safety_limits(sql_query)
            with self.redshift_engine.connect() as conn:
                result = conn.execute(text(safe_sql))
                if result.returns_rows:
                    df = pd.DataFrame(result.fetchall(), columns=result.keys())
                    return {
                        "success": True,
                        "row_count": len(df),
                        "column_count": len(df.columns),
                        "data": df.to_dict('records') if len(df) <= 100 else df.head(100).to_dict('records'),
                        "truncated": len(df) > 100,
                        "execution_time_ms": 0
                    }
                else:
                    return {
                        "success": True,
                        "message": "Query executed successfully (no results returned)",
                        "rows_affected": result.rowcount
                    }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }

    def _add_safety_limits(self, sql_query: str) -> str:
        upper_sql = sql_query.upper().strip()
        if upper_sql.startswith('SELECT') and 'LIMIT' not in upper_sql and 'COUNT(' not in upper_sql[:100]:
            sql_query += "\nLIMIT 1000"
        if any(k in upper_sql for k in ['JOIN', 'GROUP BY', 'ORDER BY', 'WINDOW']):
            sql_query = f"SET statement_timeout = '300s';\n{sql_query}"
        return sql_query

    # Additional internal helper methods go here (e.g. _extract_sql_from_response, _classify_sql_query...)

# For compatibility
RedshiftQueryAgent = EnhancedRedshiftQueryAgent