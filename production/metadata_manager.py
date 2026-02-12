"""
Advanced Metadata Management System for Production RAG System

Provides comprehensive metadata management including hierarchical document IDs,
rich metadata schemas, lifecycle management, and enterprise-grade features.
"""

import asyncio
import hashlib
import json
import logging
import time
import uuid
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Type, TypeVar, Generic
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
import sqlite3
import threading
from collections import defaultdict
import pickle
from uuid import UUID

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')

class MetadataFieldType(Enum):
    """Supported metadata field types."""
    STRING = "string"
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    DATETIME = "datetime"
    LIST = "list"
    DICT = "dict"
    GEOLOCATION = "geolocation"
    TEMPORAL = "temporal"
    CUSTOM = "custom"

class IDGenerationStrategy(Enum):
    """Document ID generation strategies."""
    UUID = "uuid"
    SEQUENTIAL = "sequential"
    CONTENT_HASH = "content_hash"
    HIERARCHICAL = "hierarchical"
    CUSTOM = "custom"

class MetadataLifecycleStage(Enum):
    """Metadata lifecycle stages."""
    DRAFT = "draft"
    ACTIVE = "active"
    ARCHIVED = "archived"
    DELETED = "deleted"

@dataclass
class FieldConstraint:
    """Constraints for metadata fields."""
    required: bool = False
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    pattern: Optional[str] = None
    allowed_values: Optional[List[Any]] = None
    unique: bool = False

@dataclass
class MetadataField:
    """Definition of a metadata field."""
    name: str
    field_type: MetadataFieldType
    description: str = ""
    constraints: FieldConstraint = field(default_factory=FieldConstraint)
    default_value: Any = None
    version: str = "1.0"
    deprecated: bool = False
    deprecation_reason: str = ""

@dataclass
class MetadataSchema:
    """Definition of a metadata schema."""
    schema_id: str
    name: str
    version: str
    description: str = ""
    fields: List[MetadataField] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    parent_schema: Optional[str] = None
    tags: List[str] = field(default_factory=list)

@dataclass
class HierarchicalDocumentID:
    """Hierarchical document ID with structured components."""
    namespace: str
    document_type: str
    version: str
    content_hash: str
    sequence: int
    timestamp: datetime
    custom_components: Dict[str, str] = field(default_factory=dict)
    
    def __str__(self) -> str:
        """Generate hierarchical ID string."""
        components = [
            self.namespace,
            self.document_type,
            self.version,
            self.content_hash[:8],  # Short hash
            str(self.sequence),
            self.timestamp.strftime("%Y%m%d%H%M%S")
        ]
        
        # Add custom components
        for key, value in self.custom_components.items():
            components.append(f"{key}:{value}")
        
        return "_".join(components)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'namespace': self.namespace,
            'document_type': self.document_type,
            'version': self.version,
            'content_hash': self.content_hash,
            'sequence': self.sequence,
            'timestamp': self.timestamp.isoformat(),
            'custom_components': self.custom_components
        }
    
    @classmethod
    def from_string(cls, id_string: str) -> 'HierarchicalDocumentID':
        """Parse hierarchical ID from string."""
        components = id_string.split('_')
        if len(components) < 6:
            raise ValueError("Invalid hierarchical ID format")
        
        return cls(
            namespace=components[0],
            document_type=components[1],
            version=components[2],
            content_hash=components[3],
            sequence=int(components[4]),
            timestamp=datetime.fromisoformat(components[5]),
            custom_components=dict(comp.split(':') for comp in components[6:] if ':' in comp)
        )

@dataclass
class MetadataRecord:
    """Record of metadata for a document."""
    document_id: str
    schema_id: str
    metadata: Dict[str, Any]
    version: int = 1
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    lifecycle_stage: MetadataLifecycleStage = MetadataLifecycleStage.ACTIVE
    created_by: Optional[str] = None
    updated_by: Optional[str] = None
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class MetadataAuditEntry:
    """Audit entry for metadata changes."""
    document_id: str
    field_name: str
    old_value: Any
    new_value: Any
    operation: str  # 'create', 'update', 'delete'
    timestamp: datetime
    user_id: Optional[str] = None
    reason: Optional[str] = None
    ip_address: Optional[str] = None

@dataclass
class MetadataRelationship:
    """Relationship between documents."""
    source_document_id: str
    target_document_id: str
    relationship_type: str
    relationship_data: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    created_by: Optional[str] = None

class MetadataValidator(ABC):
    """Abstract base class for metadata validators."""
    
    @abstractmethod
    def validate(self, value: Any, field: MetadataField) -> Tuple[bool, List[str]]:
        """Validate a metadata value against field constraints."""
        pass
    
    @abstractmethod
    def coerce(self, value: Any, field: MetadataField) -> Any:
        """Coerce a value to the correct type."""
        pass

class StringValidator(MetadataValidator):
    """Validator for string fields."""
    
    def validate(self, value: Any, field: MetadataField) -> Tuple[bool, List[str]]:
        errors = []
        
        if not isinstance(value, str):
            errors.append(f"Expected string, got {type(value).__name__}")
            return False, errors
        
        constraints = field.constraints
        
        if constraints.min_length and len(value) < constraints.min_length:
            errors.append(f"String too short (minimum {constraints.min_length} characters)")
        
        if constraints.max_length and len(value) > constraints.max_length:
            errors.append(f"String too long (maximum {constraints.max_length} characters)")
        
        if constraints.pattern and not re.match(constraints.pattern, value):
            errors.append(f"String does not match pattern: {constraints.pattern}")
        
        if constraints.allowed_values and value not in constraints.allowed_values:
            errors.append(f"Value not in allowed values: {constraints.allowed_values}")
        
        return len(errors) == 0, errors
    
    def coerce(self, value: Any, field: MetadataField) -> Any:
        return str(value)

class NumericValidator(MetadataValidator):
    """Validator for numeric fields."""
    
    def __init__(self, field_type: MetadataFieldType):
        self.field_type = field_type
    
    def validate(self, value: Any, field: MetadataField) -> Tuple[bool, List[str]]:
        errors = []
        
        if self.field_type == MetadataFieldType.INTEGER:
            if not isinstance(value, int):
                errors.append(f"Expected integer, got {type(value).__name__}")
                return False, errors
        elif self.field_type == MetadataFieldType.FLOAT:
            if not isinstance(value, (int, float)):
                errors.append(f"Expected float, got {type(value).__name__}")
                return False, errors
        
        constraints = field.constraints
        
        if constraints.min_value and value < constraints.min_value:
            errors.append(f"Value too small (minimum {constraints.min_value})")
        
        if constraints.max_value and value > constraints.max_value:
            errors.append(f"Value too large (maximum {constraints.max_value})")
        
        return len(errors) == 0, errors
    
    def coerce(self, value: Any, field: MetadataField) -> Any:
        if self.field_type == MetadataFieldType.INTEGER:
            return int(value)
        elif self.field_type == MetadataFieldType.FLOAT:
            return float(value)

class DateTimeValidator(MetadataValidator):
    """Validator for datetime fields."""
    
    def validate(self, value: Any, field: MetadataField) -> Tuple[bool, List[str]]:
        errors = []
        
        if not isinstance(value, datetime):
            errors.append(f"Expected datetime, got {type(value).__name__}")
        
        return len(errors) == 0, errors
    
    def coerce(self, value: Any, field: MetadataField) -> Any:
        if isinstance(value, str):
            return datetime.fromisoformat(value)
        return value

class MetadataValidatorFactory:
    """Factory for creating metadata validators."""
    
    @staticmethod
    def create_validator(field_type: MetadataFieldType) -> MetadataValidator:
        """Create appropriate validator for field type."""
        if field_type == MetadataFieldType.STRING:
            return StringValidator()
        elif field_type in [MetadataFieldType.INTEGER, MetadataFieldType.FLOAT]:
            return NumericValidator(field_type)
        elif field_type == MetadataFieldType.DATETIME:
            return DateTimeValidator()
        else:
            # Default validator for other types
            return MetadataValidator()

class IDGenerator(ABC):
    """Abstract base class for ID generators."""
    
    @abstractmethod
    def generate_id(self, document_content: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a document ID."""
        pass

class UUIDGenerator(IDGenerator):
    """UUID-based ID generator."""
    
    def generate_id(self, document_content: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        return str(uuid.uuid4())

class SequentialGenerator(IDGenerator):
    """Sequential ID generator."""
    
    def __init__(self, start: int = 1):
        self.counter = start
        self.lock = threading.Lock()
    
    def generate_id(self, document_content: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        with self.lock:
            id_value = f"DOC_{self.counter:08d}"
            self.counter += 1
            return id_value

class ContentHashGenerator(IDGenerator):
    """Content-based ID generator."""
    
    def generate_id(self, document_content: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        if not document_content:
            return str(uuid.uuid4())
        
        content_hash = hashlib.sha256(document_content.encode('utf-8')).hexdigest()
        return f"CH_{content_hash[:16]}"

class HierarchicalGenerator(IDGenerator):
    """Hierarchical ID generator."""
    
    def __init__(self, namespace: str = "default", document_type: str = "document"):
        self.namespace = namespace
        self.document_type = document_type
        self.sequence_counter = 1
        self.lock = threading.Lock()
    
    def generate_id(self, document_content: Optional[str] = None, 
                   metadata: Optional[Dict[str, Any]] = None) -> str:
        with self.lock:
            content_hash = "00000000"
            if document_content:
                content_hash = hashlib.sha256(document_content.encode('utf-8')).hexdigest()
            
            hierarchical_id = HierarchicalDocumentID(
                namespace=self.namespace,
                document_type=self.document_type,
                version="1.0",
                content_hash=content_hash,
                sequence=self.sequence_counter,
                timestamp=datetime.now()
            )
            
            self.sequence_counter += 1
            return str(hierarchical_id)

class MetadataManager:
    """Advanced metadata management system."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize metadata manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.db_path = config.get('metadata_db_path', 'metadata_manager.db')
        self.enable_validation = config.get('enable_validation', True)
        self.enable_auditing = config.get('enable_auditing', True)
        self.enable_relationships = config.get('enable_relationships', True)
        
        # ID generation
        self.id_strategy = config.get('id_strategy', IDGenerationStrategy.HIERARCHICAL)
        self.id_generator = self._create_id_generator()
        
        # Storage
        self.schemas = {}  # schema_id -> MetadataSchema
        self.validators = {}  # field_type -> MetadataValidator
        
        # State
        self.is_initialized = False
        self.db_lock = threading.Lock()
        
        # Initialize
        self._init_database()
        self._load_schemas()
        self._init_validators()
    
    def _create_id_generator(self) -> IDGenerator:
        """Create appropriate ID generator based on strategy."""
        if self.id_strategy == IDGenerationStrategy.UUID:
            return UUIDGenerator()
        elif self.id_strategy == IDGenerationStrategy.SEQUENTIAL:
            return SequentialGenerator()
        elif self.id_strategy == IDGenerationStrategy.CONTENT_HASH:
            return ContentHashGenerator()
        elif self.id_strategy == IDGenerationStrategy.HIERARCHICAL:
            return HierarchicalGenerator(
                namespace=self.config.get('namespace', 'default'),
                document_type=self.config.get('document_type', 'document')
            )
        else:
            return UUIDGenerator()
    
    def _init_database(self):
        """Initialize metadata database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Schemas table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_schemas (
                    schema_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    description TEXT,
                    schema_data TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    parent_schema TEXT,
                    tags TEXT
                )
            ''')
            
            # Metadata records table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_records (
                    document_id TEXT PRIMARY KEY,
                    schema_id TEXT NOT NULL,
                    metadata_data TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    updated_at TIMESTAMP NOT NULL,
                    lifecycle_stage TEXT NOT NULL,
                    created_by TEXT,
                    updated_by TEXT,
                    validation_errors TEXT
                )
            ''')
            
            # Audit trail table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    field_name TEXT NOT NULL,
                    old_value TEXT,
                    new_value TEXT,
                    operation TEXT NOT NULL,
                    timestamp TIMESTAMP NOT NULL,
                    user_id TEXT,
                    reason TEXT,
                    ip_address TEXT
                )
            ''')
            
            # Relationships table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_document_id TEXT NOT NULL,
                    target_document_id TEXT NOT NULL,
                    relationship_type TEXT NOT NULL,
                    relationship_data TEXT,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT
                )
            ''')
            
            conn.commit()
    
    def _load_schemas(self):
        """Load existing schemas from database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT schema_data FROM metadata_schemas")
            
            for row in cursor.fetchall():
                schema_data = json.loads(row[0])
                schema = MetadataSchema(**schema_data)
                self.schemas[schema.schema_id] = schema
    
    def _init_validators(self):
        """Initialize metadata validators."""
        for field_type in MetadataFieldType:
            self.validators[field_type] = MetadataValidatorFactory.create_validator(field_type)
    
    def create_schema(self, schema: MetadataSchema) -> bool:
        """Create a new metadata schema."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Check if schema already exists
                cursor.execute("SELECT schema_id FROM metadata_schemas WHERE schema_id = ?", 
                             (schema.schema_id,))
                if cursor.fetchone():
                    logger.warning(f"Schema {schema.schema_id} already exists")
                    return False
                
                # Insert schema
                cursor.execute('''
                    INSERT INTO metadata_schemas 
                    (schema_id, name, version, description, schema_data, created_at, updated_at, parent_schema, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    schema.schema_id,
                    schema.name,
                    schema.version,
                    schema.description,
                    json.dumps(asdict(schema)),
                    schema.created_at,
                    schema.updated_at,
                    schema.parent_schema,
                    json.dumps(schema.tags)
                ))
                
                conn.commit()
                self.schemas[schema.schema_id] = schema
                logger.info(f"Created schema: {schema.schema_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating schema: {e}")
            return False
    
    def get_schema(self, schema_id: str) -> Optional[MetadataSchema]:
        """Get a metadata schema by ID."""
        return self.schemas.get(schema_id)
    
    def validate_metadata(self, metadata: Dict[str, Any], schema_id: str) -> Tuple[bool, List[str]]:
        """Validate metadata against a schema."""
        if not self.enable_validation:
            return True, []
        
        schema = self.get_schema(schema_id)
        if not schema:
            return False, [f"Schema {schema_id} not found"]
        
        errors = []
        
        # Check required fields
        required_fields = {field.name for field in schema.fields if field.constraints.required}
        missing_fields = required_fields - set(metadata.keys())
        if missing_fields:
            errors.append(f"Missing required fields: {missing_fields}")
        
        # Validate each field
        for field in schema.fields:
            if field.name in metadata:
                value = metadata[field.name]
                validator = self.validators.get(field.field_type)
                
                if validator:
                    is_valid, field_errors = validator.validate(value, field)
                    if not is_valid:
                        errors.extend([f"Field {field.name}: {error}" for error in field_errors])
                    
                    # Coerce value to correct type
                    metadata[field.name] = validator.coerce(value, field)
        
        return len(errors) == 0, errors
    
    def generate_document_id(self, document_content: Optional[str] = None, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate a document ID using the configured strategy."""
        return self.id_generator.generate_id(document_content, metadata)
    
    def create_metadata_record(self, document_id: str, schema_id: str, 
                             metadata: Dict[str, Any], 
                             user_id: Optional[str] = None) -> bool:
        """Create a new metadata record."""
        try:
            # Validate metadata
            is_valid, errors = self.validate_metadata(metadata, schema_id)
            if not is_valid and self.enable_validation:
                logger.error(f"Metadata validation failed: {errors}")
                return False
            
            # Create metadata record
            record = MetadataRecord(
                document_id=document_id,
                schema_id=schema_id,
                metadata=metadata,
                validation_errors=errors if not is_valid else []
            )
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO metadata_records 
                    (document_id, schema_id, metadata_data, version, created_at, updated_at, 
                     lifecycle_stage, created_by, updated_by, validation_errors)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    record.document_id,
                    record.schema_id,
                    json.dumps(record.metadata),
                    record.version,
                    record.created_at,
                    record.updated_at,
                    record.lifecycle_stage.value,
                    user_id,
                    user_id,
                    json.dumps(record.validation_errors)
                ))
                
                conn.commit()
                logger.info(f"Created metadata record for document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating metadata record: {e}")
            return False
    
    def update_metadata_record(self, document_id: str, metadata: Dict[str, Any], 
                              user_id: Optional[str] = None, 
                              reason: Optional[str] = None) -> bool:
        """Update an existing metadata record."""
        try:
            # Get existing record
            record = self.get_metadata_record(document_id)
            if not record:
                logger.error(f"Metadata record not found for document: {document_id}")
                return False
            
            # Merge metadata
            updated_metadata = {**record.metadata, **metadata}
            
            # Validate updated metadata
            is_valid, errors = self.validate_metadata(updated_metadata, record.schema_id)
            if not is_valid and self.enable_validation:
                logger.error(f"Metadata validation failed: {errors}")
                return False
            
            # Update record
            record.metadata = updated_metadata
            record.version += 1
            record.updated_at = datetime.now()
            record.validation_errors = errors if not is_valid else []
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE metadata_records 
                    SET metadata_data = ?, version = ?, updated_at = ?, updated_by = ?, validation_errors = ?
                    WHERE document_id = ?
                ''', (
                    json.dumps(record.metadata),
                    record.version,
                    record.updated_at,
                    user_id,
                    json.dumps(record.validation_errors),
                    document_id
                ))
                
                conn.commit()
                
                # Log audit trail
                if self.enable_auditing:
                    self._log_audit_trail(document_id, metadata, user_id, reason)
                
                logger.info(f"Updated metadata record for document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error updating metadata record: {e}")
            return False
    
    def get_metadata_record(self, document_id: str) -> Optional[MetadataRecord]:
        """Get metadata record for a document."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT document_id, schema_id, metadata_data, version, created_at, 
                           updated_at, lifecycle_stage, created_by, updated_by, validation_errors
                    FROM metadata_records WHERE document_id = ?
                ''', (document_id,))
                
                row = cursor.fetchone()
                if row:
                    return MetadataRecord(
                        document_id=row[0],
                        schema_id=row[1],
                        metadata=json.loads(row[2]),
                        version=row[3],
                        created_at=datetime.fromisoformat(row[4]),
                        updated_at=datetime.fromisoformat(row[5]),
                        lifecycle_stage=MetadataLifecycleStage(row[6]),
                        created_by=row[7],
                        updated_by=row[8],
                        validation_errors=json.loads(row[9])
                    )
                return None
                
        except Exception as e:
            logger.error(f"Error getting metadata record: {e}")
            return None
    
    def delete_metadata_record(self, document_id: str, user_id: Optional[str] = None, 
                             reason: Optional[str] = None) -> bool:
        """Delete a metadata record."""
        try:
            # Archive the record before deletion
            record = self.get_metadata_record(document_id)
            if record:
                self.archive_metadata_record(document_id, user_id, reason)
            
            # Delete from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM metadata_records WHERE document_id = ?", (document_id,))
                conn.commit()
                
                logger.info(f"Deleted metadata record for document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error deleting metadata record: {e}")
            return False
    
    def archive_metadata_record(self, document_id: str, user_id: Optional[str] = None, 
                               reason: Optional[str] = None) -> bool:
        """Archive a metadata record."""
        try:
            record = self.get_metadata_record(document_id)
            if not record:
                return False
            
            # Update lifecycle stage to archived
            record.lifecycle_stage = MetadataLifecycleStage.ARCHIVED
            record.updated_at = datetime.now()
            record.updated_by = user_id
            
            # Store in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    UPDATE metadata_records 
                    SET lifecycle_stage = ?, updated_at = ?, updated_by = ?
                    WHERE document_id = ?
                ''', (
                    record.lifecycle_stage.value,
                    record.updated_at,
                    record.updated_by,
                    document_id
                ))
                
                conn.commit()
                
                # Log audit trail
                if self.enable_auditing:
                    self._log_audit_trail(document_id, {'lifecycle_stage': 'archived'}, user_id, reason)
                
                logger.info(f"Archived metadata record for document: {document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error archiving metadata record: {e}")
            return False
    
    def _log_audit_trail(self, document_id: str, changes: Dict[str, Any], 
                        user_id: Optional[str] = None, reason: Optional[str] = None):
        """Log changes to audit trail."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for field_name, new_value in changes.items():
                    # Get old value
                    old_value = None
                    record = self.get_metadata_record(document_id)
                    if record and field_name in record.metadata:
                        old_value = record.metadata[field_name]
                    
                    cursor.execute('''
                        INSERT INTO metadata_audit 
                        (document_id, field_name, old_value, new_value, operation, timestamp, user_id, reason)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        document_id,
                        field_name,
                        json.dumps(old_value) if old_value is not None else None,
                        json.dumps(new_value),
                        'update',
                        datetime.now(),
                        user_id,
                        reason
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error logging audit trail: {e}")
    
    def get_audit_trail(self, document_id: str, limit: int = 100) -> List[MetadataAuditEntry]:
        """Get audit trail for a document."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT document_id, field_name, old_value, new_value, operation, 
                           timestamp, user_id, reason
                    FROM metadata_audit 
                    WHERE document_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (document_id, limit))
                
                entries = []
                for row in cursor.fetchall():
                    entries.append(MetadataAuditEntry(
                        document_id=row[0],
                        field_name=row[1],
                        old_value=json.loads(row[2]) if row[2] else None,
                        new_value=json.loads(row[3]),
                        operation=row[4],
                        timestamp=datetime.fromisoformat(row[5]),
                        user_id=row[6],
                        reason=row[7]
                    ))
                
                return entries
                
        except Exception as e:
            logger.error(f"Error getting audit trail: {e}")
            return []
    
    def create_relationship(self, source_document_id: str, target_document_id: str, 
                          relationship_type: str, relationship_data: Optional[Dict[str, Any]] = None,
                          user_id: Optional[str] = None) -> bool:
        """Create a relationship between documents."""
        if not self.enable_relationships:
            return False
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute('''
                    INSERT INTO metadata_relationships 
                    (source_document_id, target_document_id, relationship_type, relationship_data, created_at, created_by)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    source_document_id,
                    target_document_id,
                    relationship_type,
                    json.dumps(relationship_data or {}),
                    datetime.now(),
                    user_id
                ))
                
                conn.commit()
                logger.info(f"Created relationship: {source_document_id} -> {target_document_id}")
                return True
                
        except Exception as e:
            logger.error(f"Error creating relationship: {e}")
            return False
    
    def get_relationships(self, document_id: str, relationship_type: Optional[str] = None) -> List[MetadataRelationship]:
        """Get relationships for a document."""
        if not self.enable_relationships:
            return []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                query = '''
                    SELECT source_document_id, target_document_id, relationship_type, relationship_data, created_at, created_by
                    FROM metadata_relationships 
                    WHERE source_document_id = ? OR target_document_id = ?
                '''
                params = [document_id, document_id]
                
                if relationship_type:
                    query += " AND relationship_type = ?"
                    params.append(relationship_type)
                
                cursor.execute(query, params)
                
                relationships = []
                for row in cursor.fetchall():
                    relationships.append(MetadataRelationship(
                        source_document_id=row[0],
                        target_document_id=row[1],
                        relationship_type=row[2],
                        relationship_data=json.loads(row[3]),
                        created_at=datetime.fromisoformat(row[4]),
                        created_by=row[5]
                    ))
                
                return relationships
                
        except Exception as e:
            logger.error(f"Error getting relationships: {e}")
            return []
    
    def search_metadata(self, query: Dict[str, Any], schema_id: Optional[str] = None) -> List[str]:
        """Search documents by metadata."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Build WHERE clause
                where_clauses = []
                params = []
                
                if schema_id:
                    where_clauses.append("schema_id = ?")
                    params.append(schema_id)
                
                # Add metadata search conditions
                for key, value in query.items():
                    where_clauses.append(f"json_extract(metadata_data, '$.{key}') = ?")
                    params.append(json.dumps(value))
                
                where_clause = " AND ".join(where_clauses) if where_clauses else "1=1"
                
                cursor.execute(f'''
                    SELECT document_id FROM metadata_records 
                    WHERE {where_clause}
                ''', params)
                
                return [row[0] for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error searching metadata: {e}")
            return []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get metadata management statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Document count by lifecycle stage
                cursor.execute('''
                    SELECT lifecycle_stage, COUNT(*) 
                    FROM metadata_records 
                    GROUP BY lifecycle_stage
                ''')
                lifecycle_stats = dict(cursor.fetchall())
                
                # Schema usage
                cursor.execute('''
                    SELECT schema_id, COUNT(*) 
                    FROM metadata_records 
                    GROUP BY schema_id
                ''')
                schema_stats = dict(cursor.fetchall())
                
                # Recent activity
                cursor.execute('''
                    SELECT COUNT(*) 
                    FROM metadata_audit 
                    WHERE timestamp > datetime('now', '-24 hours')
                ''')
                recent_activity = cursor.fetchone()[0]
                
                return {
                    'total_documents': sum(lifecycle_stats.values()),
                    'lifecycle_stats': lifecycle_stats,
                    'schema_usage': schema_stats,
                    'recent_activity': recent_activity,
                    'total_schemas': len(self.schemas),
                    'validation_enabled': self.enable_validation,
                    'auditing_enabled': self.enable_auditing,
                    'relationships_enabled': self.enable_relationships
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {}
    
    def export_metadata(self, document_ids: List[str], format: str = 'json') -> Dict[str, Any]:
        """Export metadata for specified documents."""
        try:
            export_data = {
                'export_date': datetime.now().isoformat(),
                'format': format,
                'documents': []
            }
            
            for document_id in document_ids:
                record = self.get_metadata_record(document_id)
                if record:
                    export_data['documents'].append({
                        'document_id': record.document_id,
                        'schema_id': record.schema_id,
                        'metadata': record.metadata,
                        'version': record.version,
                        'lifecycle_stage': record.lifecycle_stage.value,
                        'created_at': record.created_at.isoformat(),
                        'updated_at': record.updated_at.isoformat(),
                        'validation_errors': record.validation_errors
                    })
            
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting metadata: {e}")
            return {'error': str(e)}
    
    def import_metadata(self, export_data: Dict[str, Any], user_id: Optional[str] = None) -> bool:
        """Import metadata from export data."""
        try:
            for doc_data in export_data.get('documents', []):
                success = self.create_metadata_record(
                    doc_data['document_id'],
                    doc_data['schema_id'],
                    doc_data['metadata'],
                    user_id
                )
                if not success:
                    logger.error(f"Failed to import metadata for document: {doc_data['document_id']}")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing metadata: {e}")
            return False


# Convenience functions for easy use
def create_metadata_manager(config: Dict[str, Any]) -> MetadataManager:
    """Create a metadata manager instance."""
    return MetadataManager(config)


if __name__ == "__main__":
    
    # config = {
    #     'metadata_db_path': 'metadata_manager.db',
    #     'enable_validation': True,
    #     'enable_auditing': True,
    #     'enable_relationships': True,
    #     'id_strategy': IDGenerationStrategy.HIERARCHICAL,
    #     'namespace': 'enterprise',
    #     'document_type': 'document'
    # }
    
    # metadata_manager = MetadataManager(config)
    
    # Create a schema
    # schema = MetadataSchema(
    #     schema_id="document_v1",
    #     name="Document Schema",
    #     version="1.0",
    #     fields=[
    #         MetadataField("title", MetadataFieldType.STRING, constraints=FieldConstraint(required=True)),
    #         MetadataField("author", MetadataFieldType.STRING),
    #         MetadataField("created_date", MetadataFieldType.DATETIME),
    #         MetadataField("tags", MetadataFieldType.LIST)
    #     ]
    # )
    
    # metadata_manager.create_schema(schema)
    
    # Generate document ID
    # doc_id = metadata_manager.generate_document_id("document content")
    
    # Create metadata record
    # metadata = {
    #     "title": "Sample Document",
    #     "author": "John Doe",
    #     "created_date": datetime.now(),
    #     "tags": ["sample", "document"]
    # }
    
    # metadata_manager.create_metadata_record(doc_id, "document_v1", metadata)
    
    pass