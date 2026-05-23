#!/usr/bin/env python3
"""
Obsidian Vault Graph CLI - Extract and navigate relationships between markdown files
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Optional
import sys

class VaultGraphParser:
    def __init__(self, vault_path: str):
        self.vault_path = Path(vault_path)
        self.files: Dict[str, dict] = {}
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self._parse_vault()

    def _parse_vault(self):
        """Parse all markdown files in the vault"""
        for md_file in self.vault_path.rglob("*.md"):
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                relative_path = md_file.relative_to(self.vault_path)
                file_name = md_file.stem

                # Extract frontmatter
                frontmatter = self._extract_frontmatter(content)

                # Extract wikilinks [[file]] and backlinks
                wikilinks = self._extract_wikilinks(content)

                # Extract derived_from relationships from frontmatter
                derived_from = frontmatter.get('derived_from', [])
                if isinstance(derived_from, str):
                    derived_from = [derived_from]

                self.files[file_name] = {
                    'path': str(relative_path),
                    'frontmatter': frontmatter,
                    'wikilinks': wikilinks,
                    'derived_from': derived_from
                }

                # Build graph
                for link in wikilinks + derived_from:
                    clean_link = link.replace('.md', '').strip()
                    if clean_link:
                        self.graph[file_name].add(clean_link)
                        self.reverse_graph[clean_link].add(file_name)

            except Exception as e:
                print(f"Error parsing {md_file}: {e}", file=sys.stderr)

    def _extract_frontmatter(self, content: str) -> dict:
        """Extract YAML frontmatter from markdown"""
        match = re.match(r'^---\n(.*?)\n---', content, re.DOTALL)
        if match:
            try:
                fm_text = match.group(1)
                result = {}
                current_key = None
                for line in fm_text.split('\n'):
                    if line.startswith('  - '):
                        # List item
                        if current_key:
                            if current_key not in result:
                                result[current_key] = []
                            result[current_key].append(line[4:].strip())
                    elif ':' in line and not line.startswith('  '):
                        # Key-value pair
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip()
                        current_key = key
                        if value:
                            result[key] = value
                return result
            except:
                return {}
        return {}

    def _extract_wikilinks(self, content: str) -> List[str]:
        """Extract [[wikilinks]] from markdown"""
        return re.findall(r'\[\[([^\]]+)\]\]', content)

    def show_file_info(self, filename: str):
        """Show detailed info about a file"""
        if filename not in self.files:
            print(f"File '{filename}' not found")
            return

        file_info = self.files[filename]
        print(f"\n=== FILE: {filename} ===")
        print(f"Path: {file_info['path']}")
        print(f"Entity: {file_info['frontmatter'].get('entity', 'N/A')}")
        print(f"Status: {file_info['frontmatter'].get('status', 'N/A')}")

        if file_info['derived_from']:
            print(f"\n[DERIVED FROM]:")
            for dep in file_info['derived_from']:
                print(f"  -> {dep}")

        if file_info['wikilinks']:
            print(f"\n[REFERENCES]:")
            for link in file_info['wikilinks']:
                print(f"  -> {link}")

    def show_graph(self, filename: str = None, depth: int = 2, direction: str = 'forward'):
        """Show graph relationship tree"""
        if filename and filename not in self.files:
            print(f"File '{filename}' not found")
            return

        start_node = filename or list(self.files.keys())[0]
        visited = set()

        def print_tree(node, current_depth, prefix=""):
            if current_depth == 0 or node in visited:
                return
            visited.add(node)

            current_graph = self.graph if direction == 'forward' else self.reverse_graph
            neighbors = current_graph.get(node, set())

            for i, neighbor in enumerate(sorted(neighbors)):
                is_last = i == len(neighbors) - 1
                connector = "   " if is_last else " | "
                print(f"{prefix}{connector} {neighbor}")
                next_prefix = prefix + ("     " if is_last else " |   ")
                print_tree(neighbor, current_depth - 1, next_prefix)

        direction_label = "References" if direction == 'forward' else "Referenced By"
        print(f"\n[GRAPH] {start_node} ({direction_label})")
        print_tree(start_node, depth)

    def find_path(self, start: str, end: str) -> Optional[List[str]]:
        """Find path between two files using BFS"""
        from collections import deque

        if start not in self.files or end not in self.files:
            return None

        queue = deque([(start, [start])])
        visited = {start}

        while queue:
            node, path = queue.popleft()
            if node == end:
                return path

            for neighbor in self.graph.get(node, set()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None

    def list_files(self, pattern: str = None):
        """List all files, optionally filtered by pattern"""
        files = self.files.keys()
        if pattern:
            import fnmatch
            files = [f for f in files if fnmatch.fnmatch(f, pattern)]

        for f in sorted(files):
            print(f"  {f}")

    def show_stats(self):
        """Show vault statistics"""
        print("\n[STATS] Vault Statistics:")
        print(f"Total files: {len(self.files)}")
        print(f"Total relationships: {sum(len(links) for links in self.graph.values())}")

        # Find hubs (most referenced)
        most_referenced = sorted(
            [(node, len(self.reverse_graph[node])) for node in self.files.keys()],
            key=lambda x: x[1],
            reverse=True
        )[:5]

        if most_referenced:
            print(f"\nMost Referenced (Hubs):")
            for node, count in most_referenced:
                print(f"  {node}: {count} references")

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Obsidian Vault Graph CLI')
    parser.add_argument('--vault', default='vault/beliefs', help='Path to vault directory')
    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # List command
    subparsers.add_parser('list', help='List all files')

    # Info command
    info_parser = subparsers.add_parser('info', help='Show file info')
    info_parser.add_argument('file', help='Filename')

    # Graph command
    graph_parser = subparsers.add_parser('graph', help='Show graph tree')
    graph_parser.add_argument('file', nargs='?', help='Starting file')
    graph_parser.add_argument('-d', '--depth', type=int, default=2, help='Depth of tree')
    graph_parser.add_argument('-r', '--reverse', action='store_true', help='Show reverse references')

    # Path command
    path_parser = subparsers.add_parser('path', help='Find path between files')
    path_parser.add_argument('start', help='Start file')
    path_parser.add_argument('end', help='End file')

    # Stats command
    subparsers.add_parser('stats', help='Show vault statistics')

    args = parser.parse_args()

    # Initialize parser
    parser_obj = VaultGraphParser(args.vault)

    # Handle commands
    if args.command == 'list':
        parser_obj.list_files()
    elif args.command == 'info':
        parser_obj.show_file_info(args.file)
    elif args.command == 'graph':
        direction = 'reverse' if args.reverse else 'forward'
        parser_obj.show_graph(args.file, args.depth, direction)
    elif args.command == 'path':
        path = parser_obj.find_path(args.start, args.end)
        if path:
            print(f"\n[PATH] from {args.start} to {args.end}:")
            print(" -> ".join(path))
        else:
            print(f"No path found between {args.start} and {args.end}")
    elif args.command == 'stats':
        parser_obj.show_stats()
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
