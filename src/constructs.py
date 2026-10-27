"""Programming constructs per line of a Java program, found with simple, inspectable rules.

A developer profile marks each construct as known or unknown. Rules are deliberately transparent so that every
adaptation can be traced back to a line and a construct.
"""
import re

NAMES = {
    "class": "class declaration",
    "field": "field declaration",
    "constructor": "constructor",
    "this_field": "this.field (instance variable access)",
    "method_def": "method definition with return type",
    "return": "return statement",
    "conditional": "if / else",
    "main": "static main method",
    "array": "array type",
    "new": "object creation with new",
    "method_call": "method call on an object",
    "print": "printing with System.out.println",
    "arithmetic": "arithmetic expression",
}


def program_lines(source: str) -> list[str]:
    """Non-empty lines, numbered from 1 in the order they are shown."""
    return [line.strip() for line in source.split("\n") if line.strip()]


def detect(line: str, class_name: str) -> list[str]:
    found = []
    if re.search(r"\bclass\s+\w+", line):
        found.append("class")
    if re.match(r"^(private\s+|public\s+)?(int|String|double)\s+\w+\s*(,\s*\w+\s*)*;", line):
        found.append("field")
    if re.match(rf"^public\s+{class_name}\s*\(", line):
        found.append("constructor")
    if "static void main" in line:
        found.append("main")
    elif re.match(r"^public\s+(int|double|void|String)\s+\w+\s*\(", line):
        found.append("method_def")
    if "[" in line:
        found.append("array")
    if "this." in line:
        found.append("this_field")
    if re.search(r"\breturn\b", line):
        found.append("return")
    if re.search(r"\b(if|else)\b", line):
        found.append("conditional")
    if re.search(r"\bnew\s+\w+", line):
        found.append("new")
    if re.search(r"\w+\.\w+\s*\(", line) and "System.out" not in line or re.search(r"\w+\s*\(\s*\)\s*\*", line):
        found.append("method_call")
    if "System.out.println" in line:
        found.append("print")
    if re.search(r"[\w)]\s+[-+*/]\s+[\w(]", line) and "class" not in line:
        found.append("arithmetic")
    return found


def line_constructs(source: str, class_name: str) -> dict[int, list[str]]:
    return {n: detect(line, class_name) for n, line in enumerate(program_lines(source), start=1)}
