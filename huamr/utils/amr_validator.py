import penman
from amr import AMR


def format_amr(amr: str):
    return penman.encode(penman.decode(amr))


class AMRValidator:
    def __init__(self, propbank_amr_frame_arg_descr_file_path):
        self.propbank = {}
        with open(propbank_amr_frame_arg_descr_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split('  ')
                predicate = parts[0]
                arguments = {}
                for arg in parts[1:]:
                    arg_label, _, arg_description = arg.partition(': ')
                    arguments[arg_label.strip()] = arg_description
                self.propbank[predicate] = arguments

        self.validators = [
            self.validate_parentheses,
            self.validate_by_penman_parsing,
            self.validate_amr_parsing,
            self.validate_against_propbank_frames,
            self.validate_and_or_connection,
        ]

    def validate(self, amr: str) -> bool:
        amr = amr.replace('\n', ' ')

        for validator in self.validators:
            if not validator(amr):
                return False

        return True

    def validate_by_penman_parsing(self, amr: str) -> bool:
        try:
            graph = penman.decode(amr)
            penman.encode(graph)
            return True
        except Exception:
            return False

    def validate_amr_parsing(self, amr: str) -> bool:
        return AMR.parse_AMR_line(amr) is not None

    def validate_parentheses(self, amr: str) -> bool:
        if amr.count('(') == 0:
            return False
        if amr.count(')') == 0:
            return False

        inside_quotes = False
        parentheses = 0

        for char in amr:
            if char == '"':
                inside_quotes = not inside_quotes
            if not inside_quotes:
                if char == '(':
                    parentheses += 1
                elif char == ')':
                    parentheses -= 1

                    if parentheses < 0:
                        return False

        return parentheses == 0

    def validate_against_propbank_frames(self, amr: str):
        amr_graph = penman.decode(amr)

        for v1, _, predicate in amr_graph.instances():
            if predicate in self.propbank:
                allowed_args = self.propbank[predicate]

                for edge in amr_graph.edges(source=v1):
                    role = edge[1][1:]
                    if role.startswith('ARG') and role not in allowed_args:
                        return False

        return True

    def validate_and_or_connection(self, amr: str) -> bool:
        graph = penman.decode(amr)

        for v1, _, v2 in graph.instances():
            if v2 == 'and' or v2 == 'or':
                op_numbers = []
                for x1, edge, _ in graph.triples:
                    if x1 == v1 and edge.startswith(':op'):
                        op_numbers.append(int(edge[3:]))

                if len(op_numbers) == 1 and op_numbers[0] == 2:
                    return True

                if len(op_numbers) < 2:
                    return False

                if op_numbers != list(range(1, len(op_numbers) + 1)):
                    return False

        return True
