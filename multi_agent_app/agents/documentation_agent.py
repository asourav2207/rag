
"""
Documentation Agent - Specialized agent for handling documentation queries
Uses enhanced RAG with intelligent context ranking and field-specific search
"""

import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import re

# Assuming orchestrator_agent is in the same directory or a discoverable path
from .orchestrator_agent import AgentResponse, AgentType, QueryContext


class DocumentationAgent:
    """
    Specialized agent for documentation queries with enhanced retrieval and
    context understanding.
    """

    def __init__(self, rag_data, llm_client, cache_manager=None):
        self.rag_data = rag_data  # (model, row_index, ..., field_metadata)
        self.llm_client = llm_client
        self.cache_manager = cache_manager

        # Specialized prompts for different types of documentation queries
        self.prompt_templates = {
            "field_definition": """You are a technical documentation expert. The user is asking about field definitions and meanings.

Context from documentation (ranked by relevance):
{context}

User Question: {user_query}

Provide a clear, detailed explanation that:
1. Defines what the field means
2. Explains its purpose and usage
3. Mentions data type/format if available
4. Includes any business rules or constraints
5. References the source table/document

Be precise and use the exact information from the documentation.""",
            "field_comparison": """You are a technical documentation expert. The user is comparing or asking about relationships between fields.

Context from documentation (ranked by relevance):
{context}

User Question: {user_query}

Provide a comprehensive comparison that:
1. Explains each field clearly
2. Highlights similarities and differences
3. Describes any relationships between them
4. Mentions usage scenarios for each
5. References source documentation

Be analytical and structured in your response.""",
            "table_overview": """You are a technical documentation expert. The user is asking about table structure or overview.

Context from documentation (ranked by relevance):
{context}

User Question: {user_query}

Provide a comprehensive overview that:
1. Describes the table's purpose
2. Lists key fields and their meanings
3. Explains relationships to other tables
4. Mentions any important business rules
5. Provides usage examples if available

Structure your response clearly with sections.""",
            "general_documentation": """You are a technical documentation expert helping developers understand database schemas and field definitions.

Context from documentation (ranked by relevance):
{context}

User Question: {user_query}

Based STRICTLY on the provided documentation context:
1. Answer the user's question directly and accurately
2. Use specific details from the documentation
3. If multiple interpretations exist, explain them
4. Reference which document/table the information comes from
5. If information is missing, clearly state what's not available

Be precise, helpful, and cite your sources."""
        }

    async def process_query(self, query_context: QueryContext) -> AgentResponse:
        """
        Process documentation query with specialized retrieval and response generation.
        """
        start_time = datetime.now()

        try:
            # Enhanced retrieval with entity-aware search
            retrieved_docs = await self._enhanced_document_retrieval(query_context)

            if not retrieved_docs:
                return AgentResponse(
                    agent_type=AgentType.DOCUMENTATION,
                    success=False,
                    response_text="I couldn't find relevant documentation for your query. Please check if the fields or tables you're asking about exist in the documentation.",
                    context_data={},
                    confidence=0.0,
                    execution_time=(datetime.now() - start_time).total_seconds(),
                    metadata={"no_docs_found": True}
                )

            # Determine query type for specialized prompt selection
            doc_query_type = self._classify_documentation_query(query_context)

            # Generate response using specialized prompt
            response_text = await self._generate_specialized_response(
                query_context, retrieved_docs, doc_query_type
            )

            # Calculate confidence based on retrieval quality and query clarity
            confidence = self._calculate_response_confidence(
                query_context, retrieved_docs, response_text
            )

            return AgentResponse(
                agent_type=AgentType.DOCUMENTATION,
                success=True,
                response_text=response_text,
                context_data={
                    "retrieved_docs": [{"doc": doc, "metadata": meta} for doc, meta in retrieved_docs],
                    "query_type": doc_query_type,
                    "entities_found": query_context.extracted_entities
                },
                confidence=confidence,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={
                    "docs_retrieved": len(retrieved_docs),
                    "specialized_prompt": doc_query_type
                }
            )

        except Exception as e:
            return AgentResponse(
                agent_type=AgentType.DOCUMENTATION,
                success=False,
                response_text=f"I encountered an error while searching the documentation: {str(e)}",
                context_data={},
                confidence=0.0,
                execution_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )

    async def _enhanced_document_retrieval(self, query_context: QueryContext) -> List[Tuple[str, Dict]]:
        """
        Enhanced document retrieval with entity-aware search and intelligent ranking.
        """
        if not self.rag_data or not self.rag_data[0]:
            return []

        model, row_index, row_docs, row_metadata, field_index, field_docs, field_metadata = self.rag_data

        # Multi-stage retrieval approach
        all_candidates = []

        # Stage 1: Entity-specific search
        extracted_entities = query_context.extracted_entities
        if extracted_entities.get("fields"):
            for field_name in extracted_entities["fields"]:
                field_matches = await self._search_specific_field(
                    field_name, model, row_index, row_docs, row_metadata,
                    field_index, field_docs, field_metadata
                )
                all_candidates.extend(field_matches)

        # Stage 2: Semantic search with query expansion
        expanded_queries = self._expand_query_for_documentation(query_context.user_query)
        for expanded_query in expanded_queries:
            semantic_matches = await self._semantic_search(
                expanded_query, model, row_index, row_docs, row_metadata, k=5
            )
            all_candidates.extend(semantic_matches)

        # Stage 3: Keyword-based search for exact matches
        keyword_matches = await self._keyword_search(
            query_context.user_query, row_docs, row_metadata
        )
        all_candidates.extend(keyword_matches)

        # Stage 4: Intelligent deduplication and ranking
        ranked_results = self._rank_and_deduplicate_docs(all_candidates, query_context)
        return ranked_results[:8]  # Return top 8 most relevant documents

    async def _search_specific_field(
        self, field_name: str, model, row_index, row_docs, row_metadata,
        field_index, field_docs, field_metadata
    ) -> List[Tuple[str, Dict, float]]:
        """Search for specific field mentions with high precision."""
        matches = []
        field_lower = field_name.lower()

        # Direct field name search in field docs
        if field_docs and field_metadata:
            for i, (doc, meta) in enumerate(zip(field_docs, field_metadata)):
                doc_lower = doc.lower()
                if field_lower in doc_lower:
                    if doc_lower.startswith(field_lower):
                        score = 1.0  # Perfect match at start
                    elif f"field name: {field_lower}" in doc_lower:
                        score = 0.95  # Structured field name match
                    elif f"{field_lower}:" in doc_lower:
                        score = 0.9  # Field definition match
                    else:
                        score = 0.7  # General mention
                    matches.append((doc, meta, score))

        # Search in row docs for field mentions
        if row_docs and row_metadata:
            for doc, meta in zip(row_docs, row_metadata):
                if field_lower in doc.lower():
                    score = 0.6  # Lower score as less specific
                    matches.append((doc, meta, score))
        return matches

    async def _semantic_search(
        self, query: str, model, row_index, row_docs, row_metadata, k=5
    ) -> List[Tuple[str, Dict, float]]:
        """Perform semantic search using embeddings."""
        try:
            query_emb = model.encode([query], normalize_embeddings=True)
            scores, indices = row_index.search(query_emb.astype('float32'), k)
            matches = []
            for score, idx in zip(scores[0], indices[0]):
                if idx >= 0 and score > 0.1:  # Cosine similarity threshold
                    matches.append((row_docs[idx], row_metadata[idx], float(score)))
            return matches
        except Exception:
            return []

    async def _keyword_search(
        self, query: str, row_docs: List[str], row_metadata: List[Dict]
    ) -> List[Tuple[str, Dict, float]]:
        """Perform keyword-based exact matching."""
        matches = []
        query_words = set(query.lower().split())

        for doc, meta in zip(row_docs, row_metadata):
            doc_words = set(doc.lower().split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                score = overlap / len(query_words)
                if score > 0.3:  # Minimum overlap threshold
                    matches.append((doc, meta, score * 0.8))  # Lower weight
        return matches

    def _expand_query_for_documentation(self, query: str) -> List[str]:
        """Expand query with documentation-specific terms."""
        expanded = [query]  # Always include original
        query_lower = query.lower()

        if any(term in query_lower for term in ["field", "column"]):
            expanded.append(query + " definition description")
            expanded.append(query + " meaning purpose")
        if "what is" in query_lower:
            expanded.append(query.replace("what is", "definition of"))
        if any(term in query_lower for term in ["user", "claim", "payment"]):
            expanded.append(query + " business logic")
        return list(set(expanded))[:3]  # Return unique variations

    def _rank_and_deduplicate_docs(
        self, candidates: List[Tuple[str, Dict, float]],
        query_context: QueryContext
    ) -> List[Tuple[str, Dict]]:
        """Intelligent ranking and deduplication of retrieved documents."""
        doc_groups = {}
        for doc, meta, score in candidates:
            doc_key = self._generate_doc_key(doc, meta)
            if doc_key not in doc_groups:
                doc_groups[doc_key] = {
                    "doc": doc, "metadata": meta, "scores": []
                }
            doc_groups[doc_key]["scores"].append(score)

        ranked_docs = []
        for doc_key, doc_info in doc_groups.items():
            base_score = max(doc_info["scores"])
            boost_factor = 1.0

            if doc_info["metadata"].get("has_description", False):
                boost_factor += 0.2
            if doc_info["metadata"].get("field_count", 0) > 5:
                boost_factor += 0.1

            doc_text_lower = doc_info["doc"].lower()
            entities = query_context.extracted_entities
            if any(f.lower() in doc_text_lower for f in entities.get("fields", [])):
                boost_factor += 0.3
            elif any(t.lower() in doc_text_lower for t in entities.get("tables", [])):
                boost_factor += 0.2

            final_score = base_score * boost_factor
            ranked_docs.append((doc_info["doc"], doc_info["metadata"], final_score))

        ranked_docs.sort(key=lambda x: x[2], reverse=True)
        return [(doc, meta) for doc, meta, score in ranked_docs]

    def _generate_doc_key(self, doc: str, metadata: Dict) -> str:
        """Generate unique key for document deduplication."""
        return f"{metadata.get('table', 'unknown')}_{metadata.get('row_index', 0)}_{hash(doc[:100])}"

    def _classify_documentation_query(self, query_context: QueryContext) -> str:
        """Classify the type of documentation query for prompt selection."""
        query = query_context.user_query.lower()
        entities = query_context.extracted_entities

        if (any(p in query for p in ["difference", "compare", "vs", "between"]) and
                len(entities.get("fields", [])) > 1):
            return "field_comparison"
        if (any(p in query for p in ["what is", "meaning", "definition", "describe"]) and
                (entities.get("fields") or "field" in query or "column" in query)):
            return "field_definition"
        if (any(p in query for p in ["table", "overview", "structure", "schema"]) and
                entities.get("tables")):
            return "table_overview"
        return "general_documentation"

    async def _generate_specialized_response(
        self, query_context: QueryContext,
        retrieved_docs: List[Tuple[str, Dict]],
        doc_query_type: str
    ) -> str:
        """Generate response using specialized prompts."""
        context_text = self._format_context_for_llm(retrieved_docs)
        prompt_template = self.prompt_templates.get(doc_query_type, self.prompt_templates["general_documentation"])
        full_prompt = prompt_template.format(
            context=context_text, user_query=query_context.user_query
        )

        if query_context.conversation_history:
            recent_history = self._format_conversation_history(query_context.conversation_history[-2:])
            full_prompt = f"Recent conversation context:\n{recent_history}\n\n{full_prompt}"

        try:
            response = await self._call_llm_async(full_prompt)
            return response.strip()
        except Exception as e:
            return f"I found relevant documentation but encountered an error generating the response: {str(e)}"

    def _format_context_for_llm(self, retrieved_docs: List[Tuple[str, Dict]]) -> str:
        """Format retrieved documents for LLM context."""
        if not retrieved_docs:
            return "No relevant documentation found."

        formatted_context = []
        for i, (doc, meta) in enumerate(retrieved_docs[:5], 1):
            source = f"Source: {meta.get('table', 'Unknown')} (Row {meta.get('row_index', 'N/A')})"
            formatted_context.append(f"Document {i}:\n{source}\n{doc}\n")
        return "\n---\n".join(formatted_context)

    def _format_conversation_history(self, history: List) -> str:
        """Format conversation history for context."""
        if not history:
            return "No recent conversation."
        formatted = []
        for entry in history:
            if hasattr(entry, 'question') and hasattr(entry, 'answer'):
                formatted.append(f"User: {entry.question}")
                formatted.append(f"Assistant: {entry.answer[:150]}...")
        return "\n".join(formatted)

    def _calculate_response_confidence(
        self, query_context: QueryContext,
        retrieved_docs: List[Tuple[str, Dict]],
        response_text: str
    ) -> float:
        """Calculate confidence score for the response."""
        base_confidence = 0.5

        if len(retrieved_docs) >= 3:
            base_confidence += 0.2
        elif len(retrieved_docs) >= 1:
            base_confidence += 0.1

        entities = query_context.extracted_entities
        entity_matches = 0
        for doc, meta in retrieved_docs[:3]:
            doc_lower = doc.lower()
            if any(f.lower() in doc_lower for f in entities.get("fields", [])):
                entity_matches += 1
        if entity_matches > 0:
            base_confidence += 0.2

        if len(response_text) > 150 and "based on" in response_text.lower():
            base_confidence += 0.1
        if "error" in response_text.lower() or "couldn't find" in response_text.lower():
            base_confidence -= 0.4

        return min(max(base_confidence, 0.0), 1.0)

    async def _call_llm_async(self, prompt: str) -> str:
        """Async wrapper for synchronous LLM calls."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.llm_client, prompt)

