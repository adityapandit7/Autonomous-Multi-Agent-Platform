import sqlite3
import os
import re
import json
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import statistics
import logging

logger = logging.getLogger(__name__)

class DACOSDataset:
    """
    Integrates with DACOS dataset to extract real code smell patterns and thresholds.
    """
    
    def __init__(self, dacos_folder: str):
        """
        Initialize with path to DACOS folder containing SQL files and files folder.
        """
        self.dacos_folder = Path(dacos_folder)
        self.db_connection = None
        self.stats = {
            "method_lengths": [],
            "param_counts": [],
            "complexity_scores": []
        }
        
        logger.debug(f"\n📊 Loading DACOS dataset from: {dacos_folder}")
        
        if not self.dacos_folder.exists():
            logger.debug(f"   ⚠ DACOS folder not found: {dacos_folder}")
            self.thresholds = self._get_default_thresholds()
            return
        
        # Load the data
        self._load_dacos_data()
        self.thresholds = self._calculate_thresholds()
        
        logger.debug(f"   ✅ Loaded {len(self.stats['method_lengths'])} method samples")
        if "Long Method" in self.thresholds:
            logger.debug(f"   ✅ Long Method threshold: {self.thresholds['Long Method']['threshold']} lines")
        if "Long Parameter List" in self.thresholds:
            logger.debug(f"   ✅ Long Parameter threshold: {self.thresholds['Long Parameter List']['threshold']} params")
    
    def _get_default_thresholds(self):
        """Return default thresholds if DACOS loading fails."""
        return {
            "Long Method": {"threshold": 30, "severe": 50, "critical": 70, "description": "Default threshold"},
            "Long Parameter List": {"threshold": 5, "severe": 8, "critical": 12, "description": "Default threshold"},
            "Complex Conditional": {"threshold": 5, "severe": 10, "critical": 15, "description": "Default threshold"},
            "Multifaceted Abstraction": {"threshold": 2, "severe": 3, "critical": 5, "description": "Default threshold"}
        }
    
    def _load_dacos_data(self):
        """Load and analyze DACOS data from various formats."""
        
        # Try different formats
        json_loaded = self._load_from_json()
        if json_loaded and len(self.stats["method_lengths"]) > 10:
            return
            
        sql_loaded = self._load_from_sql()
        if sql_loaded and len(self.stats["method_lengths"]) > 10:
            return
        
        csv_loaded = self._load_from_csv()
        if csv_loaded and len(self.stats["method_lengths"]) > 10:
            return
        
        # If all else fails, try Java files
        if len(self.stats["method_lengths"]) < 10:
            logger.debug("   ℹ Structured data limited, trying Java files...")
            self._load_from_java_files()
    
    def _load_from_json(self) -> bool:
        """Load data from JSON files."""
        json_files = list(self.dacos_folder.glob("**/*.json"))
        
        for json_file in json_files[:5]:  # Limit to first 5 JSON files
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Handle different JSON structures
                if isinstance(data, list):
                    for item in data:
                        self._extract_from_dict(item)
                elif isinstance(data, dict):
                    if 'samples' in data:
                        for item in data['samples']:
                            self._extract_from_dict(item)
                    else:
                        self._extract_from_dict(data)
                
                logger.debug(f"   ✅ Loaded from JSON: {json_file.name}")
                return True
                
            except Exception as e:
                logger.debug(f"Error loading JSON {json_file}: {e}")
                continue
        
        return False
    
    def _load_from_csv(self) -> bool:
        """Load data from CSV files."""
        csv_files = list(self.dacos_folder.glob("**/*.csv"))
        
        for csv_file in csv_files[:5]:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        self._extract_from_dict(row)
                
                logger.debug(f"   ✅ Loaded from CSV: {csv_file.name}")
                return True
                
            except Exception as e:
                logger.debug(f"Error loading CSV {csv_file}: {e}")
                continue
        
        return False
    
    def _extract_from_dict(self, item: Dict):
        """Extract metrics from a dictionary item."""
        
        # Try to find LOC
        for key in ['loc', 'lines', 'length', 'line_count', 'num_lines']:
            if key in item:
                try:
                    val = float(item[key])
                    if 1 <= val <= 1000:
                        self.stats["method_lengths"].append(val)
                except (ValueError, TypeError):
                    pass
        
        # Try to find parameter count
        for key in ['params', 'parameters', 'param_count', 'parameter_count', 'num_params']:
            if key in item:
                try:
                    val = float(item[key])
                    if 0 <= val <= 30:
                        self.stats["param_counts"].append(val)
                except (ValueError, TypeError):
                    pass
        
        # Try to find complexity
        for key in ['complexity', 'cyclomatic', 'cognitive', 'cc']:
            if key in item:
                try:
                    val = float(item[key])
                    if 1 <= val <= 100:
                        self.stats["complexity_scores"].append(val)
                except (ValueError, TypeError):
                    pass
    
    def _load_from_sql(self) -> bool:
        """Load data from DACOS SQL files with proper error handling."""
        sql_files = [
            self.dacos_folder / "DACOSMain.sql",
            self.dacos_folder / "DACOSExtended.sql"
        ]
        
        for sql_file in sql_files:
            if not sql_file.exists():
                continue
            
            logger.debug(f"   📄 Reading SQL file: {sql_file.name}")
            
            try:
                # Create in-memory database
                conn = sqlite3.connect(':memory:')
                cursor = conn.cursor()
                
                # Read SQL file
                with open(sql_file, 'r', encoding='utf-8', errors='ignore') as f:
                    sql_content = f.read()
                
                # Split into individual statements
                statements = sql_content.split(';')
                successful = 0
                
                for i, statement in enumerate(statements):
                    statement = statement.strip()
                    if not statement:
                        continue
                    
                    try:
                        cursor.execute(statement)
                        successful += 1
                    except sqlite3.Error:
                        continue
                
                logger.debug(f"   ✅ Executed {successful} SQL statements")
                
                # Extract metrics from tables
                self._extract_metrics_from_db(cursor)
                
                conn.close()
                
                # If we got data, return success
                if len(self.stats["method_lengths"]) > 0:
                    return True
                
            except Exception as e:
                logger.debug(f"   ⚠ Error processing {sql_file.name}: {e}")
                continue
        
        return False
    
    def _extract_metrics_from_db(self, cursor: sqlite3.Cursor):
        """Extract metrics from database tables."""
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        for table in tables:
            table_name = table[0]
            
            # Skip SQLite system tables
            if table_name.startswith('sqlite_'):
                continue
            
            # Get column info
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            column_names = [col[1] for col in columns]
            
            # Look for tables with method/function data
            if any(keyword in table_name.lower() for keyword in ['method', 'function', 'metric']):
                self._extract_from_table(cursor, table_name, column_names)
    
    def _extract_from_table(self, cursor: sqlite3.Cursor, table_name: str, columns: List[str]):
        """Extract metrics from a specific table."""
        # Try to find LOC columns
        loc_keywords = ['loc', 'lines', 'length', 'line_count', 'num_lines', 'statement_count']
        for col in columns:
            if any(keyword in col.lower() for keyword in loc_keywords):
                try:
                    cursor.execute(f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL")
                    rows = cursor.fetchall()
                    for row in rows:
                        val = row[0]
                        if val is not None:
                            try:
                                num_val = float(val)
                                if 1 <= num_val <= 1000:  # Sanity check
                                    self.stats["method_lengths"].append(num_val)
                            except (ValueError, TypeError):
                                pass
                except sqlite3.Error:
                    pass
        
        # Try to find parameter count columns
        param_keywords = ['param', 'parameter', 'argument', 'num_params', 'parameter_count']
        for col in columns:
            if any(keyword in col.lower() for keyword in param_keywords):
                try:
                    cursor.execute(f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL")
                    rows = cursor.fetchall()
                    for row in rows:
                        val = row[0]
                        if val is not None:
                            try:
                                num_val = float(val)
                                if 0 <= num_val <= 30:  # Sanity check
                                    self.stats["param_counts"].append(num_val)
                            except (ValueError, TypeError):
                                pass
                except sqlite3.Error:
                    pass
        
        # Try to find complexity columns
        comp_keywords = ['complex', 'cyclomatic', 'cognitive', 'cc', 'complexity']
        for col in columns:
            if any(keyword in col.lower() for keyword in comp_keywords):
                try:
                    cursor.execute(f"SELECT {col} FROM {table_name} WHERE {col} IS NOT NULL")
                    rows = cursor.fetchall()
                    for row in rows:
                        val = row[0]
                        if val is not None:
                            try:
                                num_val = float(val)
                                if 1 <= num_val <= 100:  # Sanity check
                                    self.stats["complexity_scores"].append(num_val)
                            except (ValueError, TypeError):
                                pass
                except sqlite3.Error:
                    pass
    
    def _load_from_java_files(self):
        """Load data from Java files as fallback."""
        method_folders = [
            self.dacos_folder / "files" / "codesplit_java_method",
            self.dacos_folder / "files" / "codesplit_java_methods"
        ]
        
        for method_folder in method_folders:
            if method_folder.exists():
                java_files = list(method_folder.glob("*.java"))
                logger.debug(f"   📄 Found {len(java_files)} Java files in {method_folder.name}")
                
                for java_file in java_files[:500]:  # Limit for performance
                    try:
                        content = java_file.read_text(encoding='utf-8', errors='ignore')
                        
                        # Count lines
                        lines = len([l for l in content.split('\n') if l.strip()])
                        if 1 <= lines <= 500:
                            self.stats["method_lengths"].append(lines)
                        
                        # Count parameters
                        param_match = re.search(r'\(([^)]*)\)', content)
                        if param_match:
                            params = param_match.group(1).split(',')
                            param_count = len([p for p in params if p.strip()])
                            if param_count <= 20:
                                self.stats["param_counts"].append(param_count)
                        
                        # Simple complexity estimate
                        control_words = ['if', 'for', 'while', 'switch', 'catch']
                        complexity = sum(content.count(word) for word in control_words)
                        if complexity <= 50:
                            self.stats["complexity_scores"].append(complexity)
                            
                    except Exception:
                        continue
    
    def _calculate_thresholds(self) -> Dict:
        """Calculate thresholds from collected statistics."""
        
        thresholds = self._get_default_thresholds()
        
        # Calculate from method lengths
        if self.stats["method_lengths"] and len(self.stats["method_lengths"]) > 10:
            lengths = sorted(self.stats["method_lengths"])
            p75 = lengths[int(len(lengths) * 0.75)]
            p90 = lengths[int(len(lengths) * 0.9)]
            p95 = lengths[int(len(lengths) * 0.95)]
            
            # Python methods are typically shorter than Java
            python_factor = 0.8
            
            thresholds["Long Method"] = {
                "threshold": int(p75 * python_factor),
                "severe": int(p90 * python_factor),
                "critical": int(p95 * python_factor),
                "description": f"Based on {len(lengths)} methods",
                "samples": len(lengths)
            }
        
        # Calculate from parameter counts
        if self.stats["param_counts"] and len(self.stats["param_counts"]) > 10:
            params = sorted(self.stats["param_counts"])
            p75 = params[int(len(params) * 0.75)]
            p90 = params[int(len(params) * 0.9)]
            
            thresholds["Long Parameter List"] = {
                "threshold": int(p75),
                "severe": int(p90),
                "critical": int(p90 * 1.2),
                "description": f"Based on {len(params)} methods",
                "samples": len(params)
            }
        
        # Calculate from complexity
        if self.stats["complexity_scores"] and len(self.stats["complexity_scores"]) > 10:
            complexity = sorted(self.stats["complexity_scores"])
            p75 = complexity[int(len(complexity) * 0.75)]
            p90 = complexity[int(len(complexity) * 0.9)]
            
            thresholds["Complex Conditional"] = {
                "threshold": int(p75),
                "severe": int(p90),
                "critical": int(p90 * 1.3),
                "description": f"Based on {len(complexity)} methods",
                "samples": len(complexity)
            }
        
        # Add Multifaceted Abstraction if not present
        if "Multifaceted Abstraction" not in thresholds:
            thresholds["Multifaceted Abstraction"] = {
                "threshold": 1,
                "severe": 3,
                "critical": 5,
                "description": "Default threshold for multifaceted abstraction"
            }
        
        return thresholds
    
    def get_smell_thresholds(self) -> Dict:
        """Get code smell thresholds based on DACOS data."""
        return self.thresholds
    
    def generate_dacos_context(self) -> str:
        """Generate context string about DACOS findings for prompts."""
        
        if not hasattr(self, 'thresholds') or not self.thresholds:
            return "DACOS dataset not available. Using standard thresholds."
        
        thresholds = self.thresholds
        
        context = f"""
📊 **DACOS DATASET INSIGHTS**
================================
Based on analysis of real code from the DACOS dataset:

"""
        
        if "Long Method" in thresholds:
            lm = thresholds["Long Method"]
            context += f"• **Long Method**: Methods > {lm['threshold']} lines need refactoring\n"
            if 'severe' in lm:
                context += f"  (Severe if > {lm['severe']} lines)\n"
            if 'samples' in lm:
                context += f"  (Based on {lm['samples']} analyzed methods)\n"
        
        if "Long Parameter List" in thresholds:
            lp = thresholds["Long Parameter List"]
            context += f"\n• **Long Parameter List**: Methods with > {lp['threshold']} parameters\n"
            if 'severe' in lp:
                context += f"  (Severe if > {lp['severe']} parameters)\n"
        
        if "Complex Conditional" in thresholds:
            cc = thresholds["Complex Conditional"]
            context += f"\n• **Complex Conditional**: Complexity score > {cc['threshold']}\n"
        
        return context


# Global instance
_dacos_instance = None

def init_dacos(dacos_folder: str):
    """Initialize the DACOS dataset."""
    global _dacos_instance
    try:
        _dacos_instance = DACOSDataset(dacos_folder)
        return _dacos_instance
    except Exception as e:
        logger.debug(f"⚠ Failed to initialize DACOS: {e}")
        return None

def get_dacos():
    """Get the DACOS dataset instance."""
    return _dacos_instance