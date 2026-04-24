"""
Профилирование графа вычислений для анализа возможностей fusion.
"""

import torch
from torch.fx import symbolic_trace, GraphModule
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import warnings


@dataclass
class OperationNode:
    """Узел графа вычислений."""

    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    memory_estimate_bytes: int = 0
    compute_cost: float = 1.0  # Относительная стоимость

    @property
    def is_fusable(self) -> bool:
        """Проверяет, можно ли объединить операцию."""
        fusable_ops = [
            "add",
            "sub",
            "mul",
            "div",
            "pow",
            "sqrt",
            "abs",
            "clamp",
            "conv2d",
            "conv_transpose2d",
            "max_pool2d",
            "avg_pool2d",
            "threshold",
            "sigmoid",
            "relu",
            "softmax",
        ]
        return any(op in self.op_type.lower() for op in fusable_ops)


class GraphProfiler:
    """
    Профилировщик графа вычислений для анализа fusion potential.

    Поддерживает:
    - Трассировку через torch.fx
    - Оценку памяти и вычислительной сложности
    - Выявление фьюз-кандидатов
    - Визуализацию графа
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.profiles: Dict[str, Dict[str, Any]] = {}

    def profile_function(
        self,
        func: Callable,
        example_input: torch.Tensor,
        method_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Профилирует функцию через torch.fx.

        Args:
            func: Функция для профилирования
            example_input: Пример входа
            method_name: Название метода (для отчёта)

        Returns:
            Dict с результатами профилирования
        """
        method_name = method_name or func.__name__

        try:
            # Трассировка графа
            traced = symbolic_trace(func)
            graph = traced.graph

            # Анализ узлов
            nodes: List[OperationNode] = []
            for node in graph.nodes:
                if node.op == "call_function":
                    op_node = OperationNode(
                        name=node.name,
                        op_type=str(node.target),
                        inputs=[arg.name for arg in node.args if hasattr(arg, "name")],
                        outputs=[],  # Заполняется позже
                    )
                    nodes.append(op_node)

            # Оценка fusion potential
            fusable_count = sum(1 for n in nodes if n.is_fusable)
            fusion_potential = fusable_count / len(nodes) if nodes else 0

            # Оценка памяти
            input_memory = example_input.element_size() * example_input.numel()
            estimated_peak_memory = input_memory * (
                1 + len(nodes) * 0.1
            )  # Грубая оценка

            profile = {
                "method_name": method_name,
                "num_nodes": len(nodes),
                "num_fusable": fusable_count,
                "fusion_potential": fusion_potential,
                "estimated_peak_memory_mb": estimated_peak_memory / (1024**2),
                "nodes": nodes,
                "graph": graph,
            }

            self.profiles[method_name] = profile

            if self.verbose:
                print(
                    f"📊 {method_name}: {len(nodes)} nodes, "
                    f"{fusion_potential*100:.1f}% fusable"
                )

            return profile

        except Exception as e:
            warnings.warn(f"Profiling failed for {method_name}: {e}")
            return {
                "method_name": method_name,
                "error": str(e),
                "fusion_potential": 0,
            }

    def identify_fusion_candidates(
        self,
        profile: Dict[str, Any],
        min_chain_length: int = 3,
    ) -> List[List[str]]:
        """
        Выявляет цепочки операций для fusion.

        Args:
            profile: Результат профилирования
            min_chain_length: Мин. длина цепочки для fusion

        Returns:
            List[List[str]]: Списки имён операций для fusion
        """
        nodes = profile.get("nodes", [])
        if not nodes:
            return []

        candidates = []
        current_chain = []

        for node in nodes:
            if node.is_fusable:
                current_chain.append(node.name)
            else:
                if len(current_chain) >= min_chain_length:
                    candidates.append(current_chain.copy())
                current_chain = []

        # Проверка последней цепочки
        if len(current_chain) >= min_chain_length:
            candidates.append(current_chain)

        return candidates

    def visualize_graph(
        self,
        profile: Dict[str, Any],
        output_path: Optional[str] = None,
    ) -> Optional[str]:
        """
        Визуализирует граф вычислений.

        ⚠️ Требует graphviz для PNG вывода.
        """
        graph = profile.get("graph")
        if graph is None:
            return None

        try:
            # Текстовое представление
            graph_str = graph.print_tabular()

            if output_path and output_path.endswith(".txt"):
                with open(output_path, "w") as f:
                    f.write(graph_str)
                return output_path

            # PNG через graphviz (опционально)
            if output_path and output_path.endswith(".png"):
                try:
                    from torch.fx.graph import _type_repr
                    import graphviz

                    dot = graphviz.Digraph()
                    for node in graph.nodes:
                        dot.node(node.name, f"{node.op}\n{node.target}")
                        for arg in node.args:
                            if hasattr(arg, "name"):
                                dot.edge(arg.name, node.name)

                    dot.render(output_path.replace(".png", ""), format="png")
                    return output_path + ".png"

                except ImportError:
                    warnings.warn("graphviz not installed, saving text only")

            return None

        except Exception as e:
            warnings.warn(f"Visualization failed: {e}")
            return None
