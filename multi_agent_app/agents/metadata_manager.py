"""
Enhanced Metadata Manager for Redshift Schema
Stores schema information with generated descriptions and metadata for better SQL generation
"""

import json
import sqlite3
import pandas as pd
import re
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import hashlib
from sqlalchemy import text
import logging

class SchemaMetadataManager:
    """
    Manages Redshift schema metadata with intelligent descriptions and caching
    """

    def __init__(self, metadata_db_path: str = "redshift_metadata.db"):
        self.metadata_db_path = metadata_db_path
        self.logger = logging.getLogger(__name__)
        self._init_metadata_db()

    def _init_metadata_db(self):
        """Initialize the metadata database with required tables"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        # Create tables for metadata storage
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS schema_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            schema_name TEXT NOT NULL,
            table_name TEXT NOT NULL,
            column_name TEXT,
            data_type TEXT,
            is_nullable TEXT,
            column_default TEXT,
            ordinal_position INTEGER,
            generated_description TEXT,
            business_context TEXT,
            sample_values TEXT,
            data_quality_notes TEXT,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            schema_hash TEXT,
            UNIQUE(schema_name, table_name, column_name)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS table_relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_schema TEXT NOT NULL,
            source_table TEXT NOT NULL,
            source_column TEXT NOT NULL,
            target_schema TEXT NOT NULL,
            target_table TEXT NOT NULL,
            target_column TEXT NOT NULL,
            relationship_type TEXT,
            confidence_score REAL,
            last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(source_schema, source_table, source_column, target_schema, target_table, target_column)
        )
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS query_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_name TEXT NOT NULL,
            pattern_description TEXT,
            sql_template TEXT,
            required_tables TEXT,
            required_columns TEXT,
            usage_count INTEGER DEFAULT 0,
            last_used TIMESTAMP,
            UNIQUE(pattern_name)
        )
        """)

        conn.commit()
        conn.close()

    def extract_and_store_schema(self, redshift_engine, llm_client=None) -> Dict[str, Any]:
        """
        Extract comprehensive schema information from Redshift and store with generated descriptions
        """
        if not redshift_engine:
            return {"error": "No Redshift connection available"}

        try:
            # Extract detailed schema information
            schema_data = self._extract_detailed_schema(redshift_engine)

            # Generate descriptions for tables and columns
            if llm_client:
                schema_data = self._generate_descriptions(schema_data, llm_client)

            # Detect relationships between tables
            relationships = self._detect_table_relationships(schema_data)

            # Store in metadata database
            self._store_schema_metadata(schema_data, relationships)

            # Generate common query patterns
            query_patterns = self._generate_common_patterns(schema_data)
            self._store_query_patterns(query_patterns)

            return {
                "status": "success",
                "tables_processed": len(schema_data),
                "relationships_found": len(relationships),
                "patterns_generated": len(query_patterns),
                "schema_data": schema_data
            }

        except Exception as e:
            self.logger.error(f"Error extracting schema: {e}")
            return {"error": str(e)}

    def _extract_detailed_schema(self, redshift_engine) -> Dict[str, Any]:
        """Extract detailed schema information including constraints and statistics"""

        detailed_schema_query = """
        WITH table_info AS (
            SELECT
                t.table_schema,
                t.table_name,
                t.table_type,
                pg_stat_get_numscans(c.oid) as seq_scans,
                pg_stat_get_tuples_returned(c.oid) as tuples_returned,
                pg_relation_size(c.oid) as table_size_bytes
            FROM information_schema.tables t
            LEFT JOIN pg_class c ON c.relname = t.table_name
            WHERE t.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_internal')
        ),
        column_info AS (
            SELECT
                c.table_schema,
                c.table_name,
                c.column_name,
                c.data_type,
                c.is_nullable,
                c.column_default,
                c.ordinal_position,
                c.character_maximum_length,
                c.numeric_precision,
                c.numeric_scale
            FROM information_schema.columns c
            WHERE c.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_internal')
        ),
        constraint_info AS (
            SELECT
                kcu.table_schema,
                kcu.table_name,
                kcu.column_name,
                tc.constraint_type,
                kcu.constraint_name
            FROM information_schema.key_column_usage kcu
            JOIN information_schema.table_constraints tc
                ON kcu.constraint_name = tc.constraint_name
            WHERE kcu.table_schema NOT IN ('information_schema', 'pg_catalog', 'pg_internal')
        )
        SELECT
            ti.table_schema,
            ti.table_name,
            ti.table_type,
            ti.seq_scans,
            ti.tuples_returned,
            ti.table_size_bytes,
            ci.column_name,
            ci.data_type,
            ci.is_nullable,
            ci.column_default,
            ci.ordinal_position,
            ci.character_maximum_length,
            ci.numeric_precision,
            ci.numeric_scale,
            coni.constraint_type,
            coni.constraint_name
        FROM table_info ti
        LEFT JOIN column_info ci ON ti.table_schema = ci.table_schema AND ti.table_name = ci.table_name
        LEFT JOIN constraint_info coni ON ci.table_schema = coni.table_schema
            AND ci.table_name = coni.table_name AND ci.column_name = coni.column_name
        ORDER BY ti.table_schema, ti.table_name, ci.ordinal_position
        """

        with redshift_engine.connect() as conn:
            result = conn.execute(text(detailed_schema_query))
            schema_df = pd.DataFrame(result.fetchall(), columns=result.keys())

        # Organize schema data by table
        schema_data = {}
        for (schema, table), group in schema_df.groupby(['table_schema', 'table_name']):
            table_key = f"{schema}.{table}"

            # Get table-level information
            first_row = group.iloc[0]
            schema_data[table_key] = {
                "schema": schema,
                "table": table,
                "table_type": first_row.get('table_type'),
                "statistics": {
                    "seq_scans": first_row.get('seq_scans', 0),
                    "tuples_returned": first_row.get('tuples_returned', 0),
                    "table_size_bytes": first_row.get('table_size_bytes', 0)
                },
                "columns": []
            }

            # Add column information
            for _, row in group.iterrows():
                if pd.notna(row['column_name']):
                    column_info = {
                        "name": row['column_name'],
                        "data_type": row['data_type'],
                        "is_nullable": row['is_nullable'],
                        "column_default": row['column_default'],
                        "ordinal_position": row['ordinal_position'],
                        "character_maximum_length": row['character_maximum_length'],
                        "numeric_precision": row['numeric_precision'],
                        "numeric_scale": row['numeric_scale'],
                        "constraint_type": row['constraint_type'],
                        "constraint_name": row['constraint_name']
                    }
                    schema_data[table_key]["columns"].append(column_info)

        return schema_data

    def _generate_descriptions(self, schema_data: Dict[str, Any], llm_client) -> Dict[str, Any]:
        """Generate intelligent descriptions for tables and columns using LLM"""

        for table_key, table_info in schema_data.items():
            try:
                # Generate table description
                table_prompt = f"""
                Analyze this database table and provide a concise business description:

                Table: {table_info['table']} (Schema: {table_info['schema']})
                Type: {table_info['table_type']}

                Columns:
                {self._format_columns_for_prompt(table_info['columns'])}

                Provide a 1-2 sentence description of what this table likely represents in business terms.
                Focus on the business purpose and what kind of data it stores.
                """

                table_description = llm_client(table_prompt)
                table_info['generated_description'] = table_description.strip()

                # Generate column descriptions
                for column in table_info['columns']:
                    column_prompt = f"""
                    Analyze this database column and provide a concise description:

                    Table: {table_info['table']}
                    Column: {column['name']} ({column['data_type']})
                    Nullable: {column['is_nullable']}
                    Default: {column['column_default']}

                    Based on the column name, data type, and table context, provide a brief description
                    of what this column likely represents. Be specific about the business meaning.
                    """

                    column_description = llm_client(column_prompt)
                    column['generated_description'] = column_description.strip()

            except Exception as e:
                self.logger.warning(f"Failed to generate description for {table_key}: {e}")
                table_info['generated_description'] = f"Table containing {len(table_info['columns'])} columns"
                for column in table_info['columns']:
                    column['generated_description'] = f"{column['name']} field of type {column['data_type']}"

        return schema_data

    def _format_columns_for_prompt(self, columns: List[Dict]) -> str:
        """Format columns for LLM prompt"""
        formatted = []
        for col in columns[:10]:  # Limit to first 10 columns
            formatted.append(f"- {col['name']} ({col['data_type']}) {'NOT NULL' if col['is_nullable'] == 'NO' else 'NULLABLE'}")

        if len(columns) > 10:
            formatted.append(f"... and {len(columns) - 10} more columns")

        return "\n".join(formatted)

    def _detect_table_relationships(self, schema_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect potential relationships between tables based on column names and patterns"""
        relationships = []

        # Common foreign key patterns
        fk_patterns = [
            r'(.+)_id$',
            r'(.+)_key$',
            r'(.+)_code$',
            r'(.+)_number$'
        ]

        tables = list(schema_data.keys())

        for table_key, table_info in schema_data.items():
            for column in table_info['columns']:
                column_name = column['name'].lower()

                # Check for foreign key patterns
                for pattern in fk_patterns:
                    match = re.match(pattern, column_name)
                    if match:
                        referenced_table = match.group(1)

                        # Look for matching tables
                        for other_table_key, other_table_info in schema_data.items():
                            if other_table_key != table_key:
                                other_table_name = other_table_info['table'].lower()

                                # Check if referenced table name matches or is similar
                                if (referenced_table in other_table_name or
                                        other_table_name in referenced_table or
                                        self._calculate_similarity(referenced_table, other_table_name) > 0.8):

                                    # Look for matching primary key column
                                    for other_column in other_table_info['columns']:
                                        if (other_column['name'].lower() in ['id', f"{other_table_name}_id"] or
                                                other_column['constraint_type'] == 'PRIMARY KEY'):

                                            relationships.append({
                                                'source_schema': table_info['schema'],
                                                'source_table': table_info['table'],
                                                'source_column': column['name'],
                                                'target_schema': other_table_info['schema'],
                                                'target_table': other_table_info['table'],
                                                'target_column': other_column['name'],
                                                'relationship_type': 'FOREIGN_KEY',
                                                'confidence_score': 0.8
                                            })
                                            break
        return relationships

    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings using Levenshtein distance"""
        if len(str1) < len(str2):
            return self._calculate_similarity(str2, str1)

        if len(str2) == 0:
            return 0.0

        previous_row = list(range(len(str2) + 1))
        for i, c1 in enumerate(str1):
            current_row = [i + 1]
            for j, c2 in enumerate(str2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return 1.0 - (previous_row[-1] / len(str1))

    def _generate_common_patterns(self, schema_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate common SQL query patterns based on schema structure"""
        patterns = []

        # Basic select patterns for each table
        for table_key, table_info in schema_data.items():
            table_full_name = f"{table_info['schema']}.{table_info['table']}"

            # Basic select all
            patterns.append({
                'pattern_name': f"select_all_{table_info['table']}",
                'pattern_description': f"Select all records from {table_info['table']}",
                'sql_template': f"SELECT * FROM {table_full_name} LIMIT 100;",
                'required_tables': table_info['table'],
                'required_columns': "*"
            })

            # Count records
            patterns.append({
                'pattern_name': f"count_{table_info['table']}",
                'pattern_description': f"Count total records in {table_info['table']}",
                'sql_template': f"SELECT COUNT(*) as total_records FROM {table_full_name};",
                'required_tables': table_info['table'],
                'required_columns': "*"
            })

            # Date-based queries if date columns exist
            date_columns = [col['name'] for col in table_info['columns']
                            if 'date' in col['data_type'].lower() or 'timestamp' in col['data_type'].lower()]

            for date_col in date_columns:
                patterns.append({
                    'pattern_name': f"recent_{table_info['table']}_{date_col}",
                    'pattern_description': f"Get recent records from {table_info['table']} by {date_col}",
                    'sql_template': f"SELECT * FROM {table_full_name} WHERE {date_col} >= CURRENT_DATE - INTERVAL '30 days' ORDER BY {date_col} DESC LIMIT 100;",
                    'required_tables': table_info['table'],
                    'required_columns': f"{date_col},*"
                })

        return patterns

    def _store_schema_metadata(self, schema_data: Dict[str, Any], relationships: List[Dict[str, Any]]):
        """Store schema metadata in the database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        # Calculate schema hash for versioning
        schema_hash = hashlib.md5(json.dumps(schema_data, sort_keys=True).encode()).hexdigest()

        try:
            # Store table and column metadata
            for table_key, table_info in schema_data.items():
                for column in table_info['columns']:
                    cursor.execute("""
                    INSERT OR REPLACE INTO schema_metadata
                    (schema_name, table_name, column_name, data_type, is_nullable,
                    column_default, ordinal_position, generated_description,
                    business_context, schema_hash, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        table_info['schema'],
                        table_info['table'],
                        column['name'],
                        column['data_type'],
                        column['is_nullable'],
                        column['column_default'],
                        column['ordinal_position'],
                        column.get('generated_description', ''),
                        table_info.get('generated_description', ''),
                        schema_hash,
                        datetime.now()
                    ))

            # Store relationships
            for rel in relationships:
                cursor.execute("""
                INSERT OR REPLACE INTO table_relationships
                (source_schema, source_table, source_column, target_schema,
                target_table, target_column, relationship_type, confidence_score, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    rel['source_schema'], rel['source_table'], rel['source_column'],
                    rel['target_schema'], rel['target_table'], rel['target_column'],
                    rel['relationship_type'], rel['confidence_score'], datetime.now()
                ))

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def _store_query_patterns(self, patterns: List[Dict[str, Any]]):
        """Store query patterns in the database"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        try:
            for pattern in patterns:
                cursor.execute("""
                INSERT OR REPLACE INTO query_patterns
                (pattern_name, pattern_description, sql_template, required_tables, required_columns)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    pattern['pattern_name'],
                    pattern['pattern_description'],
                    pattern['sql_template'],
                    pattern['required_tables'],
                    pattern['required_columns']
                ))

            conn.commit()

        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    def get_enhanced_schema_for_query(self, user_query: str, extracted_entities: Dict[str, Any]) -> Dict[str, Any]:
        """Get enhanced schema information relevant to user query"""
        conn = sqlite3.connect(self.metadata_db_path)

        try:
            # Build query to find relevant tables and columns
            query_terms = user_query.lower().split()
            mentioned_tables = extracted_entities.get('tables', [])
            mentioned_fields = extracted_entities.get('fields', [])

            # Find relevant metadata
            relevant_metadata = []

            # Search by exact table/column matches
            if mentioned_tables or mentioned_fields:
                placeholders = ','.join(['?' for _ in (mentioned_tables + mentioned_fields)])
                search_query = f"""
                SELECT * FROM schema_metadata
                WHERE LOWER(table_name) IN ({placeholders})
                OR LOWER(column_name) IN ({placeholders})
                """
                df = pd.read_sql_query(search_query, conn, params=mentioned_tables + mentioned_fields)
                relevant_metadata.extend(df.to_dict('records'))

            # Fuzzy search in descriptions
            for term in query_terms:
                if len(term) > 3:  # Only search meaningful terms
                    fuzzy_query = """
                    SELECT * FROM schema_metadata
                    WHERE LOWER(generated_description) LIKE ?
                    OR LOWER(business_context) LIKE ?
                    OR LOWER(table_name) LIKE ?
                    OR LOWER(column_name) LIKE ?
                    """
                    search_term = f"%{term}%"
                    df = pd.read_sql_query(fuzzy_query, conn, params=[search_term] * 4)
                    relevant_metadata.extend(df.to_dict('records'))

            # Get relationships for relevant tables
            relevant_tables = list(set([row['table_name'] for row in relevant_metadata]))
            if relevant_tables:
                placeholders = ','.join(['?' for _ in relevant_tables])
                rel_query = f"""
                SELECT * FROM table_relationships
                WHERE source_table IN ({placeholders}) OR target_table IN ({placeholders})
                """
                rel_df = pd.read_sql_query(rel_query, conn, params=relevant_tables * 2)
                relationships = rel_df.to_dict('records')
            else:
                relationships = []

            # Get matching query patterns
            pattern_query = """
            SELECT * FROM query_patterns
            WHERE LOWER(pattern_description) LIKE ?
            ORDER BY usage_count DESC
            """
            pattern_search = f"%{' '.join(query_terms[:3])}%"
            pattern_df = pd.read_sql_query(pattern_query, conn, params=[pattern_search])
            patterns = pattern_df.to_dict('records')

            return {
                'relevant_metadata': relevant_metadata,
                'relationships': relationships,
                'suggested_patterns': patterns[:5],  # Top 5 patterns
                'total_tables_found': len(relevant_tables)
            }

        except Exception as e:
            self.logger.error(f"Error retrieving enhanced schema: {e}")
            return {'error': str(e)}
        finally:
            conn.close()

    def get_formatted_schema_for_llm(self, enhanced_schema: Dict[str, Any]) -> str:
        """Format enhanced schema information for LLM consumption"""
        if 'error' in enhanced_schema:
            return f"Error retrieving schema: {enhanced_schema['error']}"

        formatted_parts = []

        # Group metadata by table
        tables = {}
        for meta in enhanced_schema['relevant_metadata']:
            table_key = f"{meta['schema_name']}.{meta['table_name']}"
            if table_key not in tables:
                tables[table_key] = {
                    'description': meta['business_context'],
                    'columns': []
                }

            tables[table_key]['columns'].append({
                'name': meta['column_name'],
                'type': meta['data_type'],
                'nullable': meta['is_nullable'],
                'description': meta['generated_description']
            })

        # Format tables
        for table_name, table_info in tables.items():
            formatted_parts.append(f"\nTable: {table_name}")
            if table_info['description']:
                formatted_parts.append(f"Description: {table_info['description']}")

            formatted_parts.append("Columns:")
            for col in table_info['columns']:
                col_desc = f" - {col['description']}" if col['description'] else ""
                formatted_parts.append(f" - {col['name']} ({col['type']}) {'NULL' if col['nullable'] == 'YES' else 'NOT NULL'}{col_desc}")

        # Add relationships
        if enhanced_schema['relationships']:
            formatted_parts.append("\nTable Relationships:")
            for rel in enhanced_schema['relationships']:
                formatted_parts.append(
                    f" - {rel['source_table']}.{rel['source_column']} → "
                    f"{rel['target_table']}.{rel['target_column']} "
                    f"({rel['relationship_type']}, confidence: {rel['confidence_score']:.1f})"
                )

        # Add suggested patterns
        if enhanced_schema['suggested_patterns']:
            formatted_parts.append("\nSuggested Query Patterns:")
            for pattern in enhanced_schema['suggested_patterns'][:3]:
                formatted_parts.append(f" - {pattern['pattern_description']}")
                formatted_parts.append(f"   {pattern['sql_template']}")

        return "\n".join(formatted_parts)

    def update_pattern_usage(self, pattern_name: str):
        """Update usage count for a query pattern"""
        conn = sqlite3.connect(self.metadata_db_path)
        cursor = conn.cursor()

        try:
            cursor.execute("""
            UPDATE query_patterns
            SET usage_count = usage_count + 1, last_used = ?
            WHERE pattern_name = ?
            """, (datetime.now(), pattern_name))

            conn.commit()

        except Exception as e:
            self.logger.error(f"Error updating pattern usage: {e}")
        finally:
            conn.close()