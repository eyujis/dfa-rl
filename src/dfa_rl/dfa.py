from typing import Dict, List, Set, Optional
import re

class DFA:
    """
    Minimal DFA parsed from a DOT file.
    A DFA is a deterministic finite state machine that accepts/rejects
    a given string of symbols.
    Accepting states are nodes that either:
      - explicitly have shape=doublecircle, OR
      - appear in a subgraph block whose default node attr is shape=doublecircle.
    """
    def __init__(self, dot_path: str):
        self.states: List[str] = []
        self.accept_states: Set[str] = set()
        self.alphabet: List[str] = []
        self.delta: Dict[str, Dict[str, str]] = {}

        graph = self._parse_dot(dot_path)
        self._from_pydot_graph(graph)

    # --- helpers ---
    @staticmethod
    def _strip_quotes(x: Optional[str]) -> Optional[str]:
        if x is None:
            return None
        x = str(x).strip()
        if len(x) >= 2 and x[0] == x[-1] and x[0] in ("'", '"'):
            return x[1:-1]
        return x

    @staticmethod
    def _clean_dot_text(dot_text: str) -> str:
        # Allow inline '#' comments for convenience in notebooks.
        lines = [re.sub(r"#.*$", "", ln) for ln in dot_text.splitlines()]
        return "\n".join(lines) # puts a newline between each element in lines

    def _parse_dot(self, dot_path: str):
        try:
            import pydot
        except ImportError as e:
            raise ImportError("Please install pydot: pip install pydot") from e

        try:
            graphs = pydot.graph_from_dot_file(dot_path)
            if graphs:
                return graphs[0]
        except Exception:
            pass

        # open and clean the file dot_path to get the graphs
        with open(dot_path, "r") as f: # "r" is the default which opens the file dot_path for reading in text mode
            raw = f.read()
        cleaned = self._clean_dot_text(raw)
        graphs = pydot.graph_from_dot_data(cleaned)
        if graphs:
            return graphs[0]
        raise ValueError(f"Could not parse DOT file: {dot_path}")

    def _from_pydot_graph(self, G):
        # Collect top-level nodes
        node_objs = {}
        states: List[str] = []
        for n in G.get_nodes():
            name = self._strip_quotes(n.get_name())
            if name and name not in ("node", "graph", "edge"):
                states.append(name)
                node_objs[name] = n

        # Accepting from explicit node attribute (shape=doublecircle)
        accept: Set[str] = set()
        for name, n in node_objs.items():
            attrs = n.get_attributes() or {}
            shape = (self._strip_quotes(attrs.get("shape")) or "").lower()
            if shape == "doublecircle":
                accept.add(name)

        # Also check subgraphs for default node shape=doublecircle
        for sg in G.get_subgraphs():
            # Does this subgraph set node defaults to doublecircle?
            sg_accept = False
            for n in sg.get_nodes():
                if self._strip_quotes(n.get_name()) == "node":
                    shape = (self._strip_quotes((n.get_attributes() or {}).get("shape")) or "").lower()
                    if shape == "doublecircle":
                        sg_accept = True
                        break
            if sg_accept:
                # All actual nodes (not the 'node' defaults) in this subgraph are accepting
                for n in sg.get_nodes():
                    name = self._strip_quotes(n.get_name())
                    if name and name not in ("node", "graph", "edge"):
                        accept.add(name)
                        if name not in states:
                            states.append(name)

        # Edges → transitions and alphabet
        delta: Dict[str, Dict[str, str]] = {}
        alphabet: Set[str] = set()
        for e in G.get_edges():
            u = self._strip_quotes(e.get_source())
            v = self._strip_quotes(e.get_destination())
            lab = self._strip_quotes(e.get_label())
            if not u or not v or lab is None:
                continue
            alphabet.add(lab)
            delta.setdefault(u, {})[lab] = v
            if u not in states: states.append(u)
            if v not in states: states.append(v)

        # Finalize
        self.states = list(dict.fromkeys(states))
        self.accept_states = accept
        self.alphabet = sorted(alphabet)
        self.delta = delta

    # Utils
    def next_state(self, s: str, a: str) -> Optional[str]:
        return self.delta.get(s, {}).get(a)

    def is_accepting(self, s: str) -> bool:
        return s in self.accept_states

    def __repr__(self):
        return (f"FSA(states={len(self.states)}, alphabet={self.alphabet}, "
                f"accepting={sorted(self.accept_states)})")
