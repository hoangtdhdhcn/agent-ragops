"""
Advanced Versioning System for Production RAG System

Provides comprehensive version control capabilities including semantic versioning,
Git-like operations, version history management, and enterprise-grade features.
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field, fields
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Union, Callable, Type, TypeVar, Generic
from abc import ABC, abstractmethod
import sqlite3
import threading
from collections import defaultdict
import pickle
from uuid import UUID
import difflib
from enum import Enum

from config import MARKDOWN_DIR, PARENT_STORE_PATH, QDRANT_DB_PATH
from enhanced_document_manager import EnhancedDocumentManager, DocumentMetadata
from metadata_manager import MetadataManager, MetadataSchema, MetadataField, FieldConstraint, MetadataFieldType

logger = logging.getLogger(__name__)

# Type definitions
T = TypeVar('T')

class VersionStatus(Enum):
    """Version status states."""
    DRAFT = "draft"
    REVIEW = "review"
    APPROVED = "approved"
    ARCHIVED = "archived"
    DEPRECATED = "deprecated"

class VersionType(Enum):
    """Types of version changes."""
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    PRE_RELEASE = "pre_release"
    BUILD = "build"

class BranchType(Enum):
    """Types of version branches."""
    MAIN = "main"
    FEATURE = "feature"
    HOTFIX = "hotfix"
    RELEASE = "release"
    DEVELOPMENT = "development"

class ConflictResolution(Enum):
    """Conflict resolution strategies."""
    KEEP_THEIRS = "keep_theirs"
    KEEP_OURS = "keep_ours"
    MANUAL = "manual"
    MERGE = "merge"

@dataclass
class SemanticVersion:
    """Semantic version representation."""
    major: int = 0
    minor: int = 0
    patch: int = 0
    pre_release: Optional[str] = None
    build: Optional[str] = None
    
    def __str__(self) -> str:
        """Generate semantic version string."""
        version_parts = [str(self.major), str(self.minor), str(self.patch)]
        
        if self.pre_release:
            version_parts.append(f"-{self.pre_release}")
        
        if self.build:
            version_parts.append(f"+{self.build}")
        
        return ".".join(version_parts)
    
    def __lt__(self, other: 'SemanticVersion') -> bool:
        """Compare versions for ordering."""
        if self.major != other.major:
            return self.major < other.major
        if self.minor != other.minor:
            return self.minor < other.minor
        if self.patch != other.patch:
            return self.patch < other.patch
        
        # Pre-release versions have lower precedence
        if self.pre_release is None and other.pre_release is not None:
            return False
        if self.pre_release is not None and other.pre_release is None:
            return True
        if self.pre_release is not None and other.pre_release is not None:
            return self.pre_release < other.pre_release
        
        return False
    
    def bump(self, version_type: VersionType) -> 'SemanticVersion':
        """Bump version based on type."""
        if version_type == VersionType.MAJOR:
            return SemanticVersion(
                major=self.major + 1,
                minor=0,
                patch=0,
                pre_release=None,
                build=None
            )
        elif version_type == VersionType.MINOR:
            return SemanticVersion(
                major=self.major,
                minor=self.minor + 1,
                patch=0,
                pre_release=None,
                build=None
            )
        elif version_type == VersionType.PATCH:
            return SemanticVersion(
                major=self.major,
                minor=self.minor,
                patch=self.patch + 1,
                pre_release=None,
                build=None
            )
        elif version_type == VersionType.PRE_RELEASE:
            pre_release = f"alpha.{int(time.time())}"
            return SemanticVersion(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                pre_release=pre_release,
                build=None
            )
        else:  # BUILD
            build = f"build.{int(time.time())}"
            return SemanticVersion(
                major=self.major,
                minor=self.minor,
                patch=self.patch,
                pre_release=self.pre_release,
                build=build
            )
    
    @classmethod
    def parse(cls, version_string: str) -> 'SemanticVersion':
        """Parse semantic version from string."""
        pattern = r'^(\d+)\.(\d+)\.(\d+)(?:-([a-zA-Z0-9.-]+))?(?:\+([a-zA-Z0-9.-]+))?$'
        match = re.match(pattern, version_string)
        
        if not match:
            raise ValueError(f"Invalid semantic version: {version_string}")
        
        major, minor, patch = int(match.group(1)), int(match.group(2)), int(match.group(3))
        pre_release = match.group(4)
        build = match.group(5)
        
        return cls(major, minor, patch, pre_release, build)

@dataclass
class VersionBranch:
    """Version branch information."""
    branch_name: str
    branch_type: BranchType
    base_version: SemanticVersion
    created_at: datetime
    created_by: Optional[str] = None
    description: str = ""
    is_active: bool = True
    parent_branch: Optional[str] = None

@dataclass
class VersionCommit:
    """Version commit information."""
    commit_id: str
    document_id: str
    version: SemanticVersion
    branch: str
    parent_commits: List[str]
    message: str
    author: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: VersionStatus = VersionStatus.DRAFT
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_hash: Optional[str] = None
    change_summary: Optional[str] = None

@dataclass
class VersionDiff:
    """Version difference information."""
    from_version: str
    to_version: str
    diff_type: str  # 'content', 'metadata', 'both'
    changes: List[Dict[str, Any]]
    diff_summary: str
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class VersionTag:
    """Version tag information."""
    tag_name: str
    version: SemanticVersion
    commit_id: str
    created_at: datetime
    created_by: Optional[str] = None
    description: str = ""
    is_release: bool = False

@dataclass
class VersionConflict:
    """Version conflict information."""
    conflict_id: str
    document_id: str
    branch_1: str
    branch_2: str
    conflict_type: str  # 'content', 'metadata', 'both'
    conflict_details: Dict[str, Any]
    created_at: datetime
    resolved: bool = False
    resolution_strategy: Optional[ConflictResolution] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[datetime] = None

@dataclass
class VersionPermission:
    """Version permission information."""
    document_id: str
    user_id: str
    permissions: List[str]  # ['read', 'write', 'commit', 'merge', 'delete']
    branch_permissions: Dict[str, List[str]]  # branch -> permissions
    created_at: datetime
    expires_at: Optional[datetime] = None

class DiffAlgorithm(ABC):
    """Abstract base class for diff algorithms."""
    
    @abstractmethod
    def compute_diff(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """Compute difference between two content versions."""
        pass
    
    @abstractmethod
    def apply_diff(self, base_content: str, diff: Dict[str, Any]) -> str:
        """Apply diff to base content."""
        pass

class UnifiedDiffAlgorithm(DiffAlgorithm):
    """Unified diff algorithm implementation."""
    
    def compute_diff(self, old_content: str, new_content: str) -> Dict[str, Any]:
        """Compute unified diff between two content versions."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        
        diff = list(difflib.unified_diff(
            old_lines, new_lines,
            lineterm='',
            n=3  # Number of context lines
        ))
        
        return {
            'diff_type': 'unified',
            'diff_content': diff,
            'added_lines': sum(1 for line in diff if line.startswith('+') and not line.startswith('+++')),
            'removed_lines': sum(1 for line in diff if line.startswith('-') and not line.startswith('---')),
            'changed_lines': len(diff)
        }
    
    def apply_diff(self, base_content: str, diff: Dict[str, Any]) -> str:
        """Apply unified diff to base content."""
        # This is a simplified implementation
        # In practice, we want a more robust diff application
        return base_content  # Placeholder

class VersionStorage(ABC):
    """Abstract base class for version storage."""
    
    @abstractmethod
    async def store_version(self, document_id: str, version: SemanticVersion, 
                          content: str, metadata: Dict[str, Any]) -> bool:
        """Store a version of document content."""
        pass
    
    @abstractmethod
    async def retrieve_version(self, document_id: str, 
                             version: SemanticVersion) -> Optional[str]:
        """Retrieve a specific version of document content."""
        pass
    
    @abstractmethod
    async def list_versions(self, document_id: str) -> List[SemanticVersion]:
        """List all versions for a document."""
        pass
    
    @abstractmethod
    async def delete_version(self, document_id: str, version: SemanticVersion) -> bool:
        """Delete a specific version."""
        pass

class FileBasedVersionStorage(VersionStorage):
    """File-based version storage implementation."""
    
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
    
    async def store_version(self, document_id: str, version: SemanticVersion, 
                          content: str, metadata: Dict[str, Any]) -> bool:
        """Store version in file system."""
        try:
            version_dir = self.storage_path / document_id / str(version)
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Store content
            content_file = version_dir / "content.txt"
            content_file.write_text(content, encoding='utf-8')
            
            # Store metadata
            metadata_file = version_dir / "metadata.json"
            metadata_file.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            return True
            
        except Exception as e:
            logger.error(f"Error storing version: {e}")
            return False
    
    async def retrieve_version(self, document_id: str, 
                             version: SemanticVersion) -> Optional[str]:
        """Retrieve version from file system."""
        try:
            version_dir = self.storage_path / document_id / str(version)
            content_file = version_dir / "content.txt"
            
            if content_file.exists():
                return content_file.read_text(encoding='utf-8')
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving version: {e}")
            return None
    
    async def list_versions(self, document_id: str) -> List[SemanticVersion]:
        """List versions from file system."""
        try:
            doc_dir = self.storage_path / document_id
            if not doc_dir.exists():
                return []
            
            versions = []
            for version_dir in doc_dir.iterdir():
                if version_dir.is_dir():
                    try:
                        version = SemanticVersion.parse(version_dir.name)
                        versions.append(version)
                    except ValueError:
                        continue
            
            return sorted(versions)
            
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []
    
    async def delete_version(self, document_id: str, version: SemanticVersion) -> bool:
        """Delete version from file system."""
        try:
            version_dir = self.storage_path / document_id / str(version)
            if version_dir.exists():
                import shutil
                shutil.rmtree(version_dir)
            return True
            
        except Exception as e:
            logger.error(f"Error deleting version: {e}")
            return False

class VersioningSystem:
    """Advanced versioning system with Git-like operations."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize versioning system.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.db_path = config.get('versioning_db_path', 'versioning_system.db')
        self.storage_path = Path(config.get('storage_path', 'version_storage'))
        
        # Versioning settings
        self.enable_semantic_versioning = config.get('enable_semantic_versioning', True)
        self.enable_branching = config.get('enable_branching', True)
        self.enable_conflict_resolution = config.get('enable_conflict_resolution', True)
        self.enable_permissions = config.get('enable_permissions', True)
        
        # Storage
        self.storage = FileBasedVersionStorage(self.storage_path)
        self.diff_algorithm = UnifiedDiffAlgorithm()
        
        # State
        self.is_initialized = False
        self.db_lock = threading.Lock()
        
        # Initialize
        self._init_database()
        self.is_initialized = True
    
    def _init_database(self):
        """Initialize versioning database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Commits table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_commits (
                    commit_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    version TEXT NOT NULL,
                    branch TEXT NOT NULL,
                    parent_commits TEXT NOT NULL,
                    message TEXT NOT NULL,
                    author TEXT,
                    created_at TIMESTAMP NOT NULL,
                    status TEXT NOT NULL,
                    metadata TEXT,
                    file_hash TEXT,
                    change_summary TEXT
                )
            ''')
            
            # Branches table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_branches (
                    branch_name TEXT PRIMARY KEY,
                    branch_type TEXT NOT NULL,
                    base_version TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT,
                    description TEXT,
                    is_active BOOLEAN NOT NULL,
                    parent_branch TEXT
                )
            ''')
            
            # Tags table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_tags (
                    tag_name TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    commit_id TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    created_by TEXT,
                    description TEXT,
                    is_release BOOLEAN NOT NULL
                )
            ''')
            
            # Conflicts table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_conflicts (
                    conflict_id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    branch_1 TEXT NOT NULL,
                    branch_2 TEXT NOT NULL,
                    conflict_type TEXT NOT NULL,
                    conflict_details TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    resolved BOOLEAN NOT NULL,
                    resolution_strategy TEXT,
                    resolved_by TEXT,
                    resolved_at TIMESTAMP
                )
            ''')
            
            # Permissions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_permissions (
                    document_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    permissions TEXT NOT NULL,
                    branch_permissions TEXT,
                    created_at TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    PRIMARY KEY (document_id, user_id)
                )
            ''')
            
            # Diffs table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS version_diffs (
                    from_version TEXT NOT NULL,
                    to_version TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    diff_type TEXT NOT NULL,
                    changes TEXT NOT NULL,
                    diff_summary TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    PRIMARY KEY (from_version, to_version, document_id)
                )
            ''')
            
            conn.commit()
    
    async def create_commit(self, document_id: str, content: str, 
                          message: str, author: Optional[str] = None,
                          branch: str = "main", parent_commits: Optional[List[str]] = None,
                          metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Create a new version commit."""
        try:
            # Get current version
            current_version = await self.get_latest_version(document_id, branch)
            if not current_version:
                # First commit
                new_version = SemanticVersion(1, 0, 0)
            else:
                # Bump version
                new_version = current_version.bump(VersionType.MINOR)
            
            # Compute file hash
            file_hash = hashlib.sha256(content.encode('utf-8')).hexdigest()
            
            # Create commit
            commit_id = f"commit_{int(time.time())}_{hashlib.md5(f'{document_id}_{new_version}'.encode()).hexdigest()[:8]}"
            
            commit = VersionCommit(
                commit_id=commit_id,
                document_id=document_id,
                version=new_version,
                branch=branch,
                parent_commits=parent_commits or [],
                message=message,
                author=author,
                metadata=metadata or {},
                file_hash=file_hash,
                change_summary=f"Version {new_version}"
            )
            
            # Store version
            success = await self.storage.store_version(document_id, new_version, content, commit.metadata)
            if not success:
                return None
            
            # Store commit in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO version_commits 
                    (commit_id, document_id, version, branch, parent_commits, message, author, 
                     created_at, status, metadata, file_hash, change_summary)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    commit.commit_id,
                    commit.document_id,
                    str(commit.version),
                    commit.branch,
                    json.dumps(commit.parent_commits),
                    commit.message,
                    commit.author,
                    commit.created_at,
                    commit.status.value,
                    json.dumps(commit.metadata),
                    commit.file_hash,
                    commit.change_summary
                ))
                
                conn.commit()
            
            logger.info(f"Created commit {commit_id} for document {document_id} at version {new_version}")
            return commit_id
            
        except Exception as e:
            logger.error(f"Error creating commit: {e}")
            return None
    
    async def get_latest_version(self, document_id: str, branch: str = "main") -> Optional[SemanticVersion]:
        """Get the latest version for a document on a branch."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT version FROM version_commits 
                    WHERE document_id = ? AND branch = ? 
                    ORDER BY created_at DESC LIMIT 1
                ''', (document_id, branch))
                
                row = cursor.fetchone()
                if row:
                    return SemanticVersion.parse(row[0])
                return None
                
        except Exception as e:
            logger.error(f"Error getting latest version: {e}")
            return None
    
    async def get_version_content(self, document_id: str, version: SemanticVersion) -> Optional[str]:
        """Get content for a specific version."""
        return await self.storage.retrieve_version(document_id, version)
    
    async def list_versions(self, document_id: str, branch: Optional[str] = None) -> List[SemanticVersion]:
        """List all versions for a document."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if branch:
                    cursor.execute('''
                        SELECT version FROM version_commits 
                        WHERE document_id = ? AND branch = ? 
                        ORDER BY created_at DESC
                    ''', (document_id, branch))
                else:
                    cursor.execute('''
                        SELECT version FROM version_commits 
                        WHERE document_id = ? 
                        ORDER BY created_at DESC
                    ''', (document_id,))
                
                versions = []
                for row in cursor.fetchall():
                    versions.append(SemanticVersion.parse(row[0]))
                
                return sorted(versions, reverse=True)
                
        except Exception as e:
            logger.error(f"Error listing versions: {e}")
            return []
    
    async def compare_versions(self, document_id: str, version1: SemanticVersion, 
                             version2: SemanticVersion) -> Optional[VersionDiff]:
        """Compare two versions and return diff."""
        try:
            content1 = await self.get_version_content(document_id, version1)
            content2 = await self.get_version_content(document_id, version2)
            
            if not content1 or not content2:
                return None
            
            # Compute diff
            diff_data = self.diff_algorithm.compute_diff(content1, content2)
            
            version_diff = VersionDiff(
                from_version=str(version1),
                to_version=str(version2),
                diff_type='content',
                changes=[diff_data],
                diff_summary=f"Diff between {version1} and {version2}"
            )
            
            # Store diff in database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO version_diffs 
                    (from_version, to_version, document_id, diff_type, changes, diff_summary, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    str(version1),
                    str(version2),
                    document_id,
                    version_diff.diff_type,
                    json.dumps(version_diff.changes),
                    version_diff.diff_summary,
                    version_diff.created_at
                ))
                
                conn.commit()
            
            return version_diff
            
        except Exception as e:
            logger.error(f"Error comparing versions: {e}")
            return None
    
    async def create_branch(self, branch_name: str, branch_type: BranchType,
                          base_version: SemanticVersion, document_id: str,
                          description: str = "", created_by: Optional[str] = None) -> bool:
        """Create a new version branch."""
        try:
            branch = VersionBranch(
                branch_name=branch_name,
                branch_type=branch_type,
                base_version=base_version,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                parent_branch="main" if branch_name != "main" else None
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO version_branches 
                    (branch_name, branch_type, base_version, created_at, created_by, 
                     description, is_active, parent_branch)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    branch.branch_name,
                    branch.branch_type.value,
                    str(branch.base_version),
                    branch.created_at,
                    branch.created_by,
                    branch.description,
                    branch.is_active,
                    branch.parent_branch
                ))
                
                conn.commit()
            
            logger.info(f"Created branch {branch_name} for document {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating branch: {e}")
            return False
    
    async def merge_branches(self, source_branch: str, target_branch: str,
                           document_id: str, message: str, 
                           author: Optional[str] = None) -> Optional[str]:
        """Merge changes from source branch to target branch."""
        try:
            # Get latest versions from both branches
            source_version = await self.get_latest_version(document_id, source_branch)
            target_version = await self.get_latest_version(document_id, target_branch)
            
            if not source_version or not target_version:
                return None
            
            # Check for conflicts
            conflict = await self._detect_conflicts(document_id, source_version, target_version)
            if conflict:
                logger.warning(f"Conflicts detected during merge from {source_branch} to {target_branch}")
                return None
            
            # Get content from source branch
            source_content = await self.get_version_content(document_id, source_version)
            if not source_content:
                return None
            
            # Create merge commit
            merge_commit_id = await self.create_commit(
                document_id, source_content, message, author, target_branch
            )
            
            if merge_commit_id:
                logger.info(f"Merged {source_branch} into {target_branch} with commit {merge_commit_id}")
            
            return merge_commit_id
            
        except Exception as e:
            logger.error(f"Error merging branches: {e}")
            return None
    
    async def _detect_conflicts(self, document_id: str, version1: SemanticVersion, 
                              version2: SemanticVersion) -> bool:
        """Detect conflicts between two versions."""
        try:
            content1 = await self.get_version_content(document_id, version1)
            content2 = await self.get_version_content(document_id, version2)
            
            if not content1 or not content2:
                return False
            
            # Simple conflict detection - check if content is different
            return content1 != content2
            
        except Exception as e:
            logger.error(f"Error detecting conflicts: {e}")
            return False
    
    async def create_tag(self, tag_name: str, version: SemanticVersion,
                       commit_id: str, document_id: str, description: str = "",
                       created_by: Optional[str] = None, is_release: bool = False) -> bool:
        """Create a version tag."""
        try:
            tag = VersionTag(
                tag_name=tag_name,
                version=version,
                commit_id=commit_id,
                created_at=datetime.now(),
                created_by=created_by,
                description=description,
                is_release=is_release
            )
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO version_tags 
                    (tag_name, version, commit_id, created_at, created_by, description, is_release)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    tag.tag_name,
                    str(tag.version),
                    tag.commit_id,
                    tag.created_at,
                    tag.created_by,
                    tag.description,
                    tag.is_release
                ))
                
                conn.commit()
            
            logger.info(f"Created tag {tag_name} for version {version}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating tag: {e}")
            return False
    
    async def revert_to_version(self, document_id: str, target_version: SemanticVersion,
                              message: str, author: Optional[str] = None) -> Optional[str]:
        """Revert document to a previous version."""
        try:
            # Get content of target version
            target_content = await self.get_version_content(document_id, target_version)
            if not target_content:
                return None
            
            # Create revert commit
            revert_commit_id = await self.create_commit(
                document_id, target_content, message, author
            )
            
            if revert_commit_id:
                logger.info(f"Reverted {document_id} to version {target_version}")
            
            return revert_commit_id
            
        except Exception as e:
            logger.error(f"Error reverting to version: {e}")
            return None
    
    async def get_commit_history(self, document_id: str, branch: Optional[str] = None,
                               limit: int = 100) -> List[VersionCommit]:
        """Get commit history for a document."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if branch:
                    cursor.execute('''
                        SELECT commit_id, document_id, version, branch, parent_commits, message, 
                               author, created_at, status, metadata, file_hash, change_summary
                        FROM version_commits 
                        WHERE document_id = ? AND branch = ? 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (document_id, branch, limit))
                else:
                    cursor.execute('''
                        SELECT commit_id, document_id, version, branch, parent_commits, message, 
                               author, created_at, status, metadata, file_hash, change_summary
                        FROM version_commits 
                        WHERE document_id = ? 
                        ORDER BY created_at DESC 
                        LIMIT ?
                    ''', (document_id, limit))
                
                commits = []
                for row in cursor.fetchall():
                    commit = VersionCommit(
                        commit_id=row[0],
                        document_id=row[1],
                        version=SemanticVersion.parse(row[2]),
                        branch=row[3],
                        parent_commits=json.loads(row[4]),
                        message=row[5],
                        author=row[6],
                        created_at=datetime.fromisoformat(row[7]),
                        status=VersionStatus(row[8]),
                        metadata=json.loads(row[9]) if row[9] else {},
                        file_hash=row[10],
                        change_summary=row[11]
                    )
                    commits.append(commit)
                
                return commits
                
        except Exception as e:
            logger.error(f"Error getting commit history: {e}")
            return []
    
    async def get_branches(self, document_id: Optional[str] = None) -> List[VersionBranch]:
        """Get all branches for a document or all documents."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if document_id:
                    # Get branches for specific document (simplified - branches are global)
                    cursor.execute('''
                        SELECT branch_name, branch_type, base_version, created_at, created_by, 
                               description, is_active, parent_branch
                        FROM version_branches 
                        ORDER BY created_at DESC
                    ''')
                else:
                    cursor.execute('''
                        SELECT branch_name, branch_type, base_version, created_at, created_by, 
                               description, is_active, parent_branch
                        FROM version_branches 
                        ORDER BY created_at DESC
                    ''')
                
                branches = []
                for row in cursor.fetchall():
                    branch = VersionBranch(
                        branch_name=row[0],
                        branch_type=BranchType(row[1]),
                        base_version=SemanticVersion.parse(row[2]),
                        created_at=datetime.fromisoformat(row[3]),
                        created_by=row[4],
                        description=row[5],
                        is_active=row[6],
                        parent_branch=row[7]
                    )
                    branches.append(branch)
                
                return branches
                
        except Exception as e:
            logger.error(f"Error getting branches: {e}")
            return []
    
    async def get_tags(self, document_id: Optional[str] = None) -> List[VersionTag]:
        """Get all tags for a document or all documents."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                if document_id:
                    # Get tags for specific document (simplified - tags are global)
                    cursor.execute('''
                        SELECT tag_name, version, commit_id, created_at, created_by, description, is_release
                        FROM version_tags 
                        ORDER BY created_at DESC
                    ''')
                else:
                    cursor.execute('''
                        SELECT tag_name, version, commit_id, created_at, created_by, description, is_release
                        FROM version_tags 
                        ORDER BY created_at DESC
                    ''')
                
                tags = []
                for row in cursor.fetchall():
                    tag = VersionTag(
                        tag_name=row[0],
                        version=SemanticVersion.parse(row[1]),
                        commit_id=row[2],
                        created_at=datetime.fromisoformat(row[3]),
                        created_by=row[4],
                        description=row[5],
                        is_release=row[6]
                    )
                    tags.append(tag)
                
                return tags
                
        except Exception as e:
            logger.error(f"Error getting tags: {e}")
            return []
    
    async def set_version_status(self, document_id: str, version: SemanticVersion,
                               status: VersionStatus) -> bool:
        """Set status for a specific version."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    UPDATE version_commits 
                    SET status = ? 
                    WHERE document_id = ? AND version = ?
                ''', (status.value, document_id, str(version)))
                
                conn.commit()
                return cursor.rowcount > 0
                
        except Exception as e:
            logger.error(f"Error setting version status: {e}")
            return False
    
    async def cleanup_old_versions(self, document_id: str, keep_count: int = 10) -> bool:
        """Clean up old versions, keeping only the specified number."""
        try:
            versions = await self.list_versions(document_id)
            if len(versions) <= keep_count:
                return True
            
            # Keep latest versions
            versions_to_delete = versions[keep_count:]
            
            for version in versions_to_delete:
                await self.storage.delete_version(document_id, version)
            
            logger.info(f"Cleaned up {len(versions_to_delete)} old versions for {document_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get versioning system statistics."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Document count
                cursor.execute('SELECT COUNT(DISTINCT document_id) FROM version_commits')
                document_count = cursor.fetchone()[0]
                
                # Version count
                cursor.execute('SELECT COUNT(*) FROM version_commits')
                version_count = cursor.fetchone()[0]
                
                # Branch count
                cursor.execute('SELECT COUNT(*) FROM version_branches')
                branch_count = cursor.fetchone()[0]
                
                # Tag count
                cursor.execute('SELECT COUNT(*) FROM version_tags')
                tag_count = cursor.fetchone()[0]
                
                # Active branches
                cursor.execute('SELECT COUNT(*) FROM version_branches WHERE is_active = 1')
                active_branches = cursor.fetchone()[0]
                
                return {
                    'total_documents': document_count,
                    'total_versions': version_count,
                    'total_branches': branch_count,
                    'total_tags': tag_count,
                    'active_branches': active_branches,
                    'storage_path': str(self.storage_path),
                    'semantic_versioning_enabled': self.enable_semantic_versioning,
                    'branching_enabled': self.enable_branching,
                    'conflict_resolution_enabled': self.enable_conflict_resolution,
                    'permissions_enabled': self.enable_permissions
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            return {'error': str(e)}


# Convenience functions for easy use
def create_versioning_system(config: Dict[str, Any]) -> VersioningSystem:
    """Create a versioning system instance."""
    return VersioningSystem(config)


if __name__ == "__main__":
    
    # config = {
    #     'versioning_db_path': 'versioning_system.db',
    #     'storage_path': 'version_storage',
    #     'enable_semantic_versioning': True,
    #     'enable_branching': True,
    #     'enable_conflict_resolution': True,
    #     'enable_permissions': True
    # }
    
    # versioning_system = VersioningSystem(config)
    
    # Create a commit
    # commit_id = await versioning_system.create_commit(
    #     "doc_123", "Document content", "Initial commit", "user1"
    # )
    
    # Create a branch
    # await versioning_system.create_branch("feature_1", BranchType.FEATURE, 
    #                                     SemanticVersion(1, 0, 0), "doc_123")
    
    # Compare versions
    # diff = await versioning_system.compare_versions("doc_123", 
    #                                               SemanticVersion(1, 0, 0), 
    #                                               SemanticVersion(1, 1, 0))
    
    pass