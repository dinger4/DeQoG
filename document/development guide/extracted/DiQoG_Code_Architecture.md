# DiQoG 核心代码架构示例

本文件提供DiQoG关键组件的代码实现示例,帮助快速上手开发。

## 1. FSM State Controller 完整实现

```python
"""
src/core/fsm_controller.py
状态控制器 - DiQoG的核心控制逻辑
"""

from enum import Enum, auto
from typing import Dict, Any, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class SystemState(Enum):
    """系统状态定义"""
    STATE_1_UNDERSTANDING = auto()
    STATE_2_DIVERSITY_IDEATION = auto()
    STATE_3_CODE_SYNTHESIS = auto()
    STATE_4_QUALITY_VALIDATION = auto()
    STATE_5_COLLECTION = auto()
    STATE_ERROR = auto()
    STATE_COMPLETE = auto()


class TransitionAction(Enum):
    """状态转换动作"""
    TRANSITION = auto()  # 正常转换
    RETRY = auto()  # 重试当前状态
    ROLLBACK = auto()  # 回滚到之前状态
    ERROR = auto()  # 错误,无法继续


class StateController:
    """
    状态控制器
    
    职责:
    1. 管理系统状态转换
    2. 评估转换条件
    3. 处理retry和rollback机制
    4. 维护跨状态上下文
    """
    
    def __init__(self, llm_client, config):
        self.current_state = SystemState.STATE_1_UNDERSTANDING
        self.llm_client = llm_client
        self.config = config
        self.context_memory = ContextMemory()
        self.transition_history = []
        self.retry_count = {}  # 每个状态的重试次数
        
        # 初始化每个状态的重试计数
        for state in SystemState:
            self.retry_count[state] = 0
    
    def evaluate_transition(
        self, 
        current_output: Dict[str, Any], 
        tools_result: Optional[Dict[str, Any]]
    ) -> Tuple[SystemState, TransitionAction]:
        """
        评估是否满足状态转换条件
        
        Args:
            current_output: 当前状态的LLM输出
            tools_result: 工具执行结果
        
        Returns:
            (next_state, action_type)
        """
        state = self.current_state
        
        # 使用LLM作为决策器
        decision_prompt = self._construct_decision_prompt(
            state, current_output, tools_result
        )
        
        decision = self.llm_client.generate(
            decision_prompt,
            temperature=0.1  # 低温度确保决策一致性
        )
        
        # 解析决策
        action, next_state, reason = self._parse_decision(decision)
        
        # 记录转换
        self.transition_history.append({
            'from_state': state,
            'to_state': next_state,
            'action': action,
            'reason': reason,
            'timestamp': datetime.now()
        })
        
        logger.info(f"State transition decision: {state} -> {next_state} ({action})")
        logger.debug(f"Reason: {reason}")
        
        return next_state, action
    
    def _construct_decision_prompt(
        self,
        current_state: SystemState,
        current_output: Dict[str, Any],
        tools_result: Optional[Dict[str, Any]]
    ) -> str:
        """构建状态转换决策提示"""
        
        # 获取状态特定的转换条件
        conditions = self._get_state_conditions(current_state)
        
        prompt = f"""
        You are the state controller for a fault-tolerant code generation system.
        
        Current State: {current_state.name}
        
        Current Output:
        {json.dumps(current_output, indent=2)}
        
        Tools Result:
        {json.dumps(tools_result, indent=2) if tools_result else "None"}
        
        Transition Conditions for {current_state.name}:
        {conditions}
        
        Based on the current output and tools result, evaluate whether:
        1. TRANSITION: All conditions are met, proceed to next state
        2. RETRY: Some conditions not met, but recoverable (e.g., syntax error, insufficient diversity)
        3. ROLLBACK: Fundamental flaw in previous decisions (e.g., wrong approach, infeasible idea)
        4. ERROR: Unrecoverable error
        
        Respond in JSON format:
        {{
            "action": "TRANSITION|RETRY|ROLLBACK|ERROR",
            "next_state": "STATE_X_NAME",
            "reason": "Brief explanation of the decision"
        }}
        """
        
        return prompt
    
    def _get_state_conditions(self, state: SystemState) -> str:
        """获取每个状态的转换条件"""
        
        conditions_map = {
            SystemState.STATE_1_UNDERSTANDING: """
            - Problem description is fully parsed
            - Relevant knowledge is collected
            - No ambiguity in requirements
            """,
            
            SystemState.STATE_2_DIVERSITY_IDEATION: """
            - Generated ideas have sufficient diversity (SDP > threshold)
            - Ideas are feasible and executable
            - Number of ideas meets requirement (N versions)
            """,
            
            SystemState.STATE_3_CODE_SYNTHESIS: """
            - All ideas are translated to executable code
            - Code passes syntax validation
            - Implementation diversity is maintained (not too similar)
            """,
            
            SystemState.STATE_4_QUALITY_VALIDATION: """
            - All codes pass functional correctness tests
            - Test pass rate meets threshold
            - No critical bugs remain
            """,
            
            SystemState.STATE_5_COLLECTION: """
            - All N versions are collected
            - Metadata is complete
            """,
        }
        
        return conditions_map.get(state, "No specific conditions")
    
    def _parse_decision(self, decision_str: str) -> Tuple[TransitionAction, SystemState, str]:
        """解析LLM的决策输出"""
        try:
            decision = json.loads(decision_str)
            action = TransitionAction[decision['action']]
            next_state_name = decision['next_state']
            next_state = SystemState[next_state_name]
            reason = decision['reason']
            return action, next_state, reason
        except Exception as e:
            logger.error(f"Failed to parse decision: {e}")
            return TransitionAction.ERROR, SystemState.STATE_ERROR, str(e)
    
    def execute_transition(
        self, 
        next_state: SystemState, 
        carry_forward_data: Dict[str, Any]
    ):
        """
        执行状态转换
        
        Args:
            next_state: 目标状态
            carry_forward_data: 需要携带到下一状态的数据
        """
        logger.info(f"Executing transition: {self.current_state} -> {next_state}")
        
        # 持久化当前状态的关键信息
        self.context_memory.persist_to_next_state(
            self.current_state,
            next_state,
            carry_forward_data
        )
        
        # 更新状态
        old_state = self.current_state
        self.current_state = next_state
        
        # 重置新状态的重试计数
        self.retry_count[next_state] = 0
        
        logger.info(f"State transitioned successfully: {old_state} -> {next_state}")
    
    def handle_retry(self, error_info: Dict[str, Any]):
        """
        处理可恢复的错误 - 重试当前状态
        
        Args:
            error_info: 错误信息,用于更新prompt
        """
        self.retry_count[self.current_state] += 1
        
        if self.retry_count[self.current_state] > self.config.max_retries:
            logger.warning(f"Max retries exceeded for {self.current_state}")
            # 触发rollback或error
            self.handle_rollback(
                self._get_previous_state(),
                "Max retries exceeded"
            )
            return
        
        logger.info(
            f"Retry {self.retry_count[self.current_state]} "
            f"for state {self.current_state}"
        )
        
        # 更新上下文,包含错误信息用于下次尝试
        self.context_memory.add_feedback(self.current_state, error_info)
    
    def handle_rollback(
        self, 
        target_state: SystemState, 
        failure_reason: str
    ):
        """
        处理根本性缺陷 - 回滚到之前的状态
        
        Args:
            target_state: 回滚目标状态
            failure_reason: 失败原因
        """
        logger.warning(
            f"Rollback triggered: {self.current_state} -> {target_state}"
        )
        logger.warning(f"Reason: {failure_reason}")
        
        # 记录失败信息,避免重复同样的错误
        self.context_memory.add_rollback_info(
            from_state=self.current_state,
            to_state=target_state,
            reason=failure_reason
        )
        
        # 执行回滚
        self.current_state = target_state
        self.retry_count[target_state] = 0
    
    def _get_previous_state(self) -> SystemState:
        """获取前一个状态"""
        state_order = [
            SystemState.STATE_1_UNDERSTANDING,
            SystemState.STATE_2_DIVERSITY_IDEATION,
            SystemState.STATE_3_CODE_SYNTHESIS,
            SystemState.STATE_4_QUALITY_VALIDATION,
            SystemState.STATE_5_COLLECTION
        ]
        
        try:
            current_idx = state_order.index(self.current_state)
            if current_idx > 0:
                return state_order[current_idx - 1]
            else:
                return SystemState.STATE_ERROR
        except ValueError:
            return SystemState.STATE_ERROR
    
    def get_context(self) -> Dict[str, Any]:
        """获取当前完整上下文"""
        return {
            'current_state': self.current_state,
            'memory': self.context_memory.get_all(),
            'transition_history': self.transition_history,
            'retry_counts': self.retry_count
        }


class ContextMemory:
    """
    上下文记忆管理
    维护跨状态的信息传递
    """
    
    def __init__(self):
        self.task_context = {}
        self.generation_history = []
        self.feedback_accumulation = []
        self.tool_outputs_cache = {}
        self.rollback_info = []
    
    def update_context(self, state: SystemState, data: Dict[str, Any]):
        """更新特定状态的上下文"""
        if state not in self.task_context:
            self.task_context[state] = []
        self.task_context[state].append({
            'timestamp': datetime.now(),
            'data': data
        })
    
    def get_state_context(self, state: SystemState) -> Dict[str, Any]:
        """获取特定状态需要的上下文"""
        return {
            'task_info': self.task_context.get(state, []),
            'history': self.generation_history,
            'feedback': self.feedback_accumulation,
            'tool_outputs': self.tool_outputs_cache,
            'rollback_warnings': self.rollback_info
        }
    
    def persist_to_next_state(
        self,
        source_state: SystemState,
        target_state: SystemState,
        data: Dict[str, Any]
    ):
        """在状态转换时持久化关键信息"""
        self.generation_history.append({
            'from': source_state,
            'to': target_state,
            'data': data,
            'timestamp': datetime.now()
        })
    
    def add_feedback(self, state: SystemState, feedback: Dict[str, Any]):
        """添加执行反馈"""
        self.feedback_accumulation.append({
            'state': state,
            'feedback': feedback,
            'timestamp': datetime.now()
        })
    
    def add_rollback_info(
        self,
        from_state: SystemState,
        to_state: SystemState,
        reason: str
    ):
        """记录回滚信息"""
        self.rollback_info.append({
            'from': from_state,
            'to': to_state,
            'reason': reason,
            'timestamp': datetime.now()
        })
    
    def get_all(self) -> Dict[str, Any]:
        """获取所有上下文"""
        return {
            'task_context': self.task_context,
            'generation_history': self.generation_history,
            'feedback': self.feedback_accumulation,
            'tool_outputs': self.tool_outputs_cache,
            'rollback_info': self.rollback_info
        }
```

---

## 2. Diversity Enhancing Agent 完整实现

```python
"""
src/agents/diversity_agent.py
多样性增强Agent - 实现HILE和IRQN算法
"""

from typing import List, Dict, Any
import random
import logging

logger = logging.getLogger(__name__)


class DiversityEnhancingAgent:
    """
    State 2: 多样性增强Agent
    
    核心算法:
    1. HILE (Hierarchical Isolation and Local Expansion)
    2. IRQN (Iterative Retention, Questioning and Negation)
    """
    
    def __init__(
        self,
        llm_client,
        diversity_evaluator,
        knowledge_search,
        dynamic_prompt_generator,
        config
    ):
        self.llm_client = llm_client
        self.diversity_evaluator = diversity_evaluator
        self.knowledge_search = knowledge_search
        self.prompt_generator = dynamic_prompt_generator
        self.config = config
        
        # HILE参数
        self.num_thoughts = config.diversity.hile.num_thoughts
        self.num_solutions = config.diversity.hile.num_solutions
        self.num_implementations = config.diversity.hile.num_implementations
        
        # IRQN参数
        self.p_qn1 = config.diversity.irqn.p_qn1
        self.p_qn2 = config.diversity.irqn.p_qn2
        self.max_iterations = config.diversity.irqn.max_iterations
        self.theta_diff = config.diversity.irqn.theta_diff
        self.theta_ident = config.diversity.irqn.theta_ident
    
    def process(
        self,
        understanding_result: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理State 2: 生成多层次多样化思路
        
        Returns:
            {
                'thought_level': [...],
                'solution_level': [...],
                'implementation_level': [...],
                'diversity_scores': {...}
            }
        """
        logger.info("Starting diversity ideation (State 2)")
        
        # 提取任务信息
        task_info = understanding_result['task_understanding']
        knowledge = understanding_result.get('collected_knowledge', {})
        
        # Level 1: Thought-level diversity (算法思路)
        logger.info("Generating thought-level diversity...")
        thoughts = self.explore_thought_level(task_info, knowledge)
        
        # 应用IRQN增强思路多样性
        thoughts = self.apply_irqn(thoughts, 'thought', knowledge)
        
        # Level 2: Solution-level diversity (伪代码方案)
        logger.info("Generating solution-level diversity...")
        solutions = self.explore_solution_level(thoughts, knowledge)
        
        # 应用IRQN增强方案多样性
        solutions = self.apply_irqn(solutions, 'solution', knowledge)
        
        # Level 3: Implementation-level diversity (具体实现)
        logger.info("Generating implementation-level diversity...")
        implementations = self.explore_implementation_level(solutions, knowledge)
        
        # 应用IRQN增强实现多样性
        implementations = self.apply_irqn(implementations, 'implementation', knowledge)
        
        # 计算多样性分数
        diversity_scores = self._calculate_diversity_scores({
            'thoughts': thoughts,
            'solutions': solutions,
            'implementations': implementations
        })
        
        logger.info(f"Diversity scores: {diversity_scores}")
        
        return {
            'thought_level': thoughts,
            'solution_level': solutions,
            'implementation_level': implementations,
            'diversity_scores': diversity_scores
        }
    
    def explore_thought_level(
        self,
        task_info: Dict[str, Any],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        思想层探索
        生成不同的算法思路 (自然语言描述)
        
        策略:
        1. 检索算法模式知识库
        2. 生成多个不同算法paradigm的思路
        3. 确保覆盖不同的time/space complexity tradeoffs
        """
        thoughts = []
        
        # 检索相关算法模式
        algo_patterns = self.knowledge_search.execute({
            'query': task_info['problem_description'],
            'knowledge_type': 'algorithmic'
        })
        
        # 生成多个思路
        for i in range(self.num_thoughts):
            prompt = self.prompt_generator.execute({
                'state': 'STATE_2_THOUGHT_GENERATION',
                'task_info': task_info,
                'knowledge': knowledge,
                'algo_patterns': algo_patterns,
                'iteration': i,
                'existing_thoughts': thoughts
            })
            
            thought = self.llm_client.generate(prompt, temperature=0.8)
            
            thoughts.append({
                'id': f'thought_{i}',
                'content': thought,
                'type': 'algorithmic_approach',
                'meta': {
                    'iteration': i,
                    'paradigm': self._extract_paradigm(thought)
                }
            })
        
        return thoughts
    
    def explore_solution_level(
        self,
        thoughts: List[Dict[str, Any]],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        方案层探索
        为每个思路生成伪代码级别的实现策略
        """
        solutions = []
        
        for thought in thoughts:
            # 为每个思路生成多个伪代码方案
            for j in range(self.num_solutions):
                prompt = self.prompt_generator.execute({
                    'state': 'STATE_2_SOLUTION_GENERATION',
                    'thought': thought,
                    'knowledge': knowledge,
                    'iteration': j
                })
                
                solution = self.llm_client.generate(prompt, temperature=0.7)
                
                solutions.append({
                    'id': f'solution_{thought["id"]}_{j}',
                    'parent_thought': thought['id'],
                    'content': solution,
                    'type': 'pseudocode_strategy',
                    'meta': {
                        'iteration': j,
                        'data_structures': self._extract_data_structures(solution)
                    }
                })
        
        return solutions
    
    def explore_implementation_level(
        self,
        solutions: List[Dict[str, Any]],
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        实现层探索
        为每个方案生成具体的代码实现方案
        
        考虑:
        - 数据结构选择 (list vs array vs dict)
        - 控制流模式 (recursion vs iteration)
        - 库函数调用 (standard lib vs custom)
        """
        implementations = []
        
        for solution in solutions:
            # 为每个方案生成多个实现变体
            for k in range(self.num_implementations):
                prompt = self.prompt_generator.execute({
                    'state': 'STATE_2_IMPLEMENTATION_PLANNING',
                    'solution': solution,
                    'knowledge': knowledge,
                    'iteration': k,
                    'variation_hints': self._get_variation_hints(k)
                })
                
                implementation = self.llm_client.generate(prompt, temperature=0.6)
                
                implementations.append({
                    'id': f'impl_{solution["id"]}_{k}',
                    'parent_solution': solution['id'],
                    'content': implementation,
                    'type': 'implementation_scheme',
                    'meta': {
                        'iteration': k,
                        'variant': self._classify_variant(implementation)
                    }
                })
        
        return implementations
    
    def apply_irqn(
        self,
        outputs: List[Dict[str, Any]],
        level: str,
        knowledge: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        应用IRQN方法增强多样性
        
        Iterative Retention, Questioning and Negation:
        - Retain: 保留完全不同的输出
        - Question: 质疑部分相似的输出并refine
        - Negate: 否定过于相似的输出并重新生成
        """
        logger.info(f"Applying IRQN to {level} level outputs")
        
        final_outputs = []
        pending_outputs = outputs.copy()
        history = []
        
        for iteration in range(self.max_iterations):
            logger.debug(f"IRQN iteration {iteration + 1}")
            
            current_batch = []
            
            for output in pending_outputs:
                # 概率触发判断
                if random.random() > self.p_qn1:
                    # 直接接受
                    final_outputs.append(output)
                    logger.debug(f"Direct accept: {output['id']}")
                    continue
                
                # 评估相似度
                similarity = self._evaluate_similarity(
                    output,
                    history + final_outputs,
                    level
                )
                
                logger.debug(f"Similarity for {output['id']}: {similarity:.3f}")
                
                # 根据相似度决定操作
                if similarity < self.theta_diff:
                    # Retain: 完全不同
                    if random.random() < self.p_qn2:
                        # 进一步否定以产生更多多样性
                        negated = self._negate_and_regenerate(
                            output, knowledge, level
                        )
                        current_batch.append(negated)
                        logger.debug(f"Retain + Negate: {output['id']}")
                    else:
                        final_outputs.append(output)
                        logger.debug(f"Retain: {output['id']}")
                
                elif similarity <= self.theta_ident:
                    # Question: 部分相似
                    questioned = self._question_and_refine(
                        output, knowledge, level
                    )
                    current_batch.append(questioned)
                    logger.debug(f"Question: {output['id']}")
                
                else:
                    # Negate: 过于相似
                    negated = self._negate_and_regenerate(
                        output, knowledge, level
                    )
                    current_batch.append(negated)
                    logger.debug(f"Negate: {output['id']}")
            
            pending_outputs = current_batch
            history.extend(final_outputs)
            
            if not pending_outputs:
                logger.info(f"IRQN converged at iteration {iteration + 1}")
                break
        
        logger.info(f"IRQN completed: {len(final_outputs)} outputs retained")
        return final_outputs
    
    def _evaluate_similarity(
        self,
        output: Dict[str, Any],
        reference_set: List[Dict[str, Any]],
        level: str
    ) -> float:
        """评估输出与参考集的相似度"""
        if not reference_set:
            return 0.0
        
        similarities = []
        for ref in reference_set:
            if level == 'thought':
                sim = self.diversity_evaluator.calculate_semantic_similarity(
                    output['content'],
                    ref['content']
                )
            elif level == 'solution':
                sim = self.diversity_evaluator.calculate_methodological_difference(
                    output['content'],
                    ref['content']
                )
            else:  # implementation
                sim = self.diversity_evaluator.calculate_semantic_similarity(
                    output['content'],
                    ref['content']
                )
            similarities.append(sim)
        
        return max(similarities)
    
    def _question_and_refine(
        self,
        output: Dict[str, Any],
        knowledge: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """质疑并细化输出"""
        prompt = f"""
        Current {level}-level output:
        {output['content']}
        
        This output shows partial similarity with existing solutions.
        
        Please refine it by:
        1. Identifying what makes it similar to existing approaches
        2. Proposing alternative approaches that are more distinctive
        3. Enhancing its uniqueness while maintaining feasibility
        
        Knowledge base context:
        {json.dumps(knowledge, indent=2)}
        
        Provide the refined {level}-level solution:
        """
        
        refined_content = self.llm_client.generate(prompt, temperature=0.7)
        
        return {
            'id': f'{output["id"]}_refined',
            'parent': output['id'],
            'content': refined_content,
            'type': output['type'],
            'meta': {
                **output.get('meta', {}),
                'refined': True,
                'original_similarity': output.get('similarity', 0)
            }
        }
    
    def _negate_and_regenerate(
        self,
        output: Dict[str, Any],
        knowledge: Dict[str, Any],
        level: str
    ) -> Dict[str, Any]:
        """否定当前输出并重新生成"""
        prompt = f"""
        Current {level}-level output:
        {output['content']}
        
        This approach has been used or is too similar to existing solutions.
        
        Please generate a COMPLETELY DIFFERENT solution that:
        1. Uses a different {'algorithmic paradigm' if level=='thought' else 'implementation strategy'}
        2. Employs different {'approach' if level=='thought' else 'data structures and control flow'}
        3. Takes a contrasting perspective to solve the problem
        
        IMPORTANT: Avoid any similarity to the current output.
        
        Knowledge base context:
        {json.dumps(knowledge, indent=2)}
        
        Provide the new {level}-level solution:
        """
        
        regenerated_content = self.llm_client.generate(prompt, temperature=0.9)
        
        return {
            'id': f'{output["id"]}_negated',
            'parent': output['id'],
            'content': regenerated_content,
            'type': output['type'],
            'meta': {
                **output.get('meta', {}),
                'negated': True,
                'original_rejected': True
            }
        }
    
    def _calculate_diversity_scores(
        self,
        all_outputs: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, float]:
        """计算各层次的多样性分数"""
        scores = {}
        
        for level, outputs in all_outputs.items():
            if not outputs:
                scores[level] = 0.0
                continue
            
            contents = [o['content'] for o in outputs]
            
            # 计算MBCS (语义相似度)
            mbcs = self.diversity_evaluator.compute_mbcs(contents)
            
            # 计算SDP (方法论差异)
            sdp = self.diversity_evaluator.compute_sdp(
                contents,
                self.llm_client
            )
            
            scores[f'{level}_mbcs'] = mbcs
            scores[f'{level}_sdp'] = sdp
        
        return scores
    
    def _extract_paradigm(self, thought: str) -> str:
        """从思路中提取算法范式"""
        # 简化实现,实际可以更复杂
        paradigms = ['dynamic_programming', 'greedy', 'divide_conquer', 
                    'backtracking', 'graph', 'sorting', 'searching']
        for paradigm in paradigms:
            if paradigm.replace('_', ' ') in thought.lower():
                return paradigm
        return 'other'
    
    def _extract_data_structures(self, solution: str) -> List[str]:
        """从方案中提取数据结构"""
        ds = []
        ds_keywords = {
            'array': ['array', 'list'],
            'hash_table': ['dict', 'hash', 'map'],
            'tree': ['tree', 'bst'],
            'graph': ['graph', 'node', 'edge'],
            'queue': ['queue'],
            'stack': ['stack'],
            'heap': ['heap', 'priority']
        }
        
        solution_lower = solution.lower()
        for ds_type, keywords in ds_keywords.items():
            if any(kw in solution_lower for kw in keywords):
                ds.append(ds_type)
        
        return ds
    
    def _classify_variant(self, implementation: str) -> str:
        """分类实现变体"""
        impl_lower = implementation.lower()
        if 'recursive' in impl_lower or 'recursion' in impl_lower:
            return 'recursive'
        elif 'iterative' in impl_lower or 'iteration' in impl_lower or 'loop' in impl_lower:
            return 'iterative'
        else:
            return 'mixed'
    
    def _get_variation_hints(self, iteration: int) -> List[str]:
        """获取变体提示"""
        hints = [
            ['Use iterative approach', 'Prefer list comprehension'],
            ['Use recursive approach', 'Optimize with memoization'],
            ['Use built-in functions', 'Minimize custom code'],
            ['Use custom implementations', 'Avoid built-ins for control']
        ]
        return hints[iteration % len(hints)]
```

---

## 3. 使用示例

```python
"""
examples/basic_usage.py
DiQoG基本使用示例
"""

import yaml
from diqog import DiQoGPipeline, Config

def main():
    # 1. 加载配置
    config = Config.from_yaml('configs/default_config.yaml')
    
    # 2. 初始化pipeline
    pipeline = DiQoGPipeline(config)
    
    # 3. 定义编程任务
    task_description = """
    Design and implement a function to find the longest palindromic substring in a given string.
    
    Requirements:
    - Input: A string s (1 <= len(s) <= 1000)
    - Output: The longest palindromic substring
    - If multiple palindromes have the same maximum length, return the first one
    
    Function signature:
    def longest_palindrome(s: str) -> str:
        pass
    
    Examples:
    Input: "babad"
    Output: "bab" (or "aba" is also valid)
    
    Input: "cbbd"
    Output: "bb"
    """
    
    # 4. 定义测试用例
    test_cases = [
        {'input': 'babad', 'expected_output': 'bab'},
        {'input': 'cbbd', 'expected_output': 'bb'},
        {'input': 'a', 'expected_output': 'a'},
        {'input': 'ac', 'expected_output': 'a'},
        {'input': 'racecar', 'expected_output': 'racecar'},
    ]
    
    # 5. 生成N版本代码
    print("Starting N-version code generation...")
    result = pipeline.generate_n_versions(
        task_description=task_description,
        test_cases=test_cases,
        n=5  # 生成5个版本
    )
    
    # 6. 输出结果
    print(f"\n{'='*60}")
    print(f"Generation Results")
    print(f"{'='*60}")
    
    print(f"\nGenerated {len(result['n_version_codes'])} diverse implementations")
    
    print(f"\nDiversity Metrics:")
    for metric, value in result['diversity_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    print(f"\nQuality Metrics:")
    for metric, value in result['quality_metrics'].items():
        print(f"  {metric}: {value:.4f}")
    
    # 7. 显示生成的代码
    print(f"\n{'='*60}")
    print(f"Generated Code Versions")
    print(f"{'='*60}")
    
    for i, code_info in enumerate(result['n_version_codes'], 1):
        print(f"\n--- Version {i} ---")
        print(f"Algorithm: {code_info['meta'].get('algorithm', 'Unknown')}")
        print(f"Pass Rate: {code_info['metrics']['pass_rate']:.2%}")
        print(f"\nCode:\n{code_info['code'][:200]}...")  # 显示前200字符
    
    # 8. 运行故障注入实验
    print(f"\n{'='*60}")
    print(f"Running Fault Injection Experiments")
    print(f"{'='*60}")
    
    from diqog.experiments import FaultInjectionExperiment
    
    fi_experiment = FaultInjectionExperiment(n_versions=5)
    fi_results = fi_experiment.run_experiment(
        n_version_codes=[c['code'] for c in result['n_version_codes']],
        test_cases=test_cases,
        patterns={
            'code_level': ['Pat-CL 0', 'Pat-CL 1', 'Pat-CL 3'],
            'algorithm_level': []  # 暂不测试算法级
        }
    )
    
    print(f"\nFault Injection Results:")
    for pattern, metrics in fi_results['code_level'].items():
        print(f"\n{pattern}:")
        print(f"  Failure Rate: {metrics['failure_rate']:.2%}")
        print(f"  Majority Consensus Rate: {metrics['mcr']:.2%}")
        print(f"  Complete Consensus Rate: {metrics['ccr']:.2%}")

if __name__ == '__main__':
    main()
```

---

## 4. 配置文件完整示例

```yaml
# configs/default_config.yaml

# LLM配置
llm:
  model_name: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  api_base: "https://api.openai.com/v1"
  temperature: 0.7
  max_tokens: 2000
  timeout: 60

# 多样性配置
diversity:
  threshold: 0.6
  hile:
    num_thoughts: 5
    num_solutions: 3
    num_implementations: 2
  irqn:
    p_qn1: 0.7
    p_qn2: 0.3
    max_iterations: 5
    theta_diff: 0.3
    theta_ident: 0.7

# 质量保证配置
quality:
  threshold: 0.9
  max_refinement_iterations: 5
  test_timeout: 5

# FSM配置
fsm:
  max_retries: 3
  enable_rollback: true
  decision_temperature: 0.1

# N版本配置
n_versions:
  default_n: 5
  min_n: 3
  max_n: 10

# 工具配置
tools:
  code_interpreter:
    timeout: 5
    sandbox_enabled: true
  diversity_evaluator:
    model_name: "microsoft/codebert-base"
    similarity_threshold: 0.7
  test_executor:
    parallel: true
    max_workers: 4

# 知识库配置
knowledge_bases:
  algorithmic_patterns: "data/knowledge_bases/algorithmic_patterns.json"
  implementation_techniques: "data/knowledge_bases/implementation_techniques.json"
  fault_tolerance_strategies: "data/knowledge_bases/fault_tolerance_strategies.json"

# 数据集配置
datasets:
  root_dir: "data/datasets"
  available:
    - "MBPP"
    - "HumanEval"
    - "ClassEval"
    - "MIPD"

# 实验配置
experiments:
  fault_injection:
    code_level_patterns: 
      - "Pat-CL 0"
      - "Pat-CL 1"
      - "Pat-CL 2"
      - "Pat-CL 3"
      - "Pat-CL 4"
    algorithm_level_patterns:
      - "Pat-AL 0"
      - "Pat-AL 1"
      - "Pat-AL 2"
      - "Pat-AL 3"
      - "Pat-AL 4"
  ablation:
    variants:
      - "DiQoG-Full"
      - "DiQoG-NoDiv"
      - "DiQoG-NoQA"

# 日志配置
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/diqog.log"
  console: true

# 缓存配置
cache:
  enabled: true
  directory: ".cache"
  max_size_mb: 1000
```

---

## 5. 测试用例示例

```python
"""
tests/test_diversity_agent.py
Diversity Agent单元测试
"""

import unittest
from unittest.mock import Mock, patch
from diqog.agents import DiversityEnhancingAgent

class TestDiversityEnhancingAgent(unittest.TestCase):
    
    def setUp(self):
        """测试设置"""
        self.mock_llm = Mock()
        self.mock_diversity_evaluator = Mock()
        self.mock_knowledge_search = Mock()
        self.mock_prompt_generator = Mock()
        self.mock_config = self._create_mock_config()
        
        self.agent = DiversityEnhancingAgent(
            llm_client=self.mock_llm,
            diversity_evaluator=self.mock_diversity_evaluator,
            knowledge_search=self.mock_knowledge_search,
            dynamic_prompt_generator=self.mock_prompt_generator,
            config=self.mock_config
        )
    
    def _create_mock_config(self):
        """创建mock配置"""
        config = Mock()
        config.diversity.hile.num_thoughts = 3
        config.diversity.hile.num_solutions = 2
        config.diversity.hile.num_implementations = 2
        config.diversity.irqn.p_qn1 = 0.7
        config.diversity.irqn.p_qn2 = 0.3
        config.diversity.irqn.max_iterations = 3
        config.diversity.irqn.theta_diff = 0.3
        config.diversity.irqn.theta_ident = 0.7
        return config
    
    def test_explore_thought_level(self):
        """测试思想层探索"""
        task_info = {
            'problem_description': 'Find the longest palindromic substring'
        }
        knowledge = {'algorithmic': ['dynamic_programming', 'two_pointers']}
        
        # Mock knowledge search
        self.mock_knowledge_search.execute.return_value = {
            'patterns': ['expansion_around_center', 'dynamic_programming']
        }
        
        # Mock prompt generator
        self.mock_prompt_generator.execute.return_value = "Generate algorithm..."
        
        # Mock LLM generation
        self.mock_llm.generate.side_effect = [
            "Use dynamic programming approach",
            "Use expansion around center",
            "Use Manacher's algorithm"
        ]
        
        # 执行
        thoughts = self.agent.explore_thought_level(task_info, knowledge)
        
        # 验证
        self.assertEqual(len(thoughts), 3)
        self.assertEqual(thoughts[0]['type'], 'algorithmic_approach')
        self.assertIn('paradigm', thoughts[0]['meta'])
        
        # 验证调用
        self.assertEqual(self.mock_llm.generate.call_count, 3)
    
    def test_apply_irqn_retain(self):
        """测试IRQN的retain逻辑"""
        outputs = [
            {'id': 'test_1', 'content': 'Approach A', 'type': 'thought'},
            {'id': 'test_2', 'content': 'Approach B', 'type': 'thought'},
        ]
        knowledge = {}
        
        # Mock similarity evaluation (完全不同)
        self.mock_diversity_evaluator.calculate_semantic_similarity.return_value = 0.1
        
        # 执行IRQN
        with patch('random.random', return_value=0.8):  # > p_qn1, 直接接受
            result = self.agent.apply_irqn(outputs, 'thought', knowledge)
        
        # 验证: 应该直接保留
        self.assertEqual(len(result), 2)
    
    def test_apply_irqn_negate(self):
        """测试IRQN的negate逻辑"""
        outputs = [
            {'id': 'test_1', 'content': 'Similar approach', 'type': 'thought'},
        ]
        knowledge = {}
        
        # Mock similarity evaluation (过于相似)
        self.mock_diversity_evaluator.calculate_semantic_similarity.return_value = 0.9
        
        # Mock LLM regeneration
        self.mock_llm.generate.return_value = "Completely different approach"
        
        # 执行IRQN
        with patch('random.random', return_value=0.5):  # < p_qn1, 触发判断
            result = self.agent.apply_irqn(outputs, 'thought', knowledge)
        
        # 验证: 应该重新生成
        self.assertTrue(any(o['meta'].get('negated') for o in result))
    
    def test_diversity_scores_calculation(self):
        """测试多样性分数计算"""
        all_outputs = {
            'thoughts': [
                {'content': 'DP approach'},
                {'content': 'Two pointers'},
                {'content': 'Manacher'}
            ],
            'solutions': [
                {'content': 'Solution 1'},
                {'content': 'Solution 2'}
            ],
            'implementations': [
                {'content': 'Impl 1'},
                {'content': 'Impl 2'}
            ]
        }
        
        # Mock diversity metrics
        self.mock_diversity_evaluator.compute_mbcs.return_value = 0.4
        self.mock_diversity_evaluator.compute_sdp.return_value = 0.7
        
        # 执行
        scores = self.agent._calculate_diversity_scores(all_outputs)
        
        # 验证
        self.assertIn('thoughts_mbcs', scores)
        self.assertIn('thoughts_sdp', scores)
        self.assertEqual(scores['thoughts_mbcs'], 0.4)
        self.assertEqual(scores['thoughts_sdp'], 0.7)

if __name__ == '__main__':
    unittest.main()
```

---

这份代码架构示例提供了DiQoG核心组件的详细实现,包括:

1. ✅ **StateController**: 完整的FSM状态控制逻辑
2. ✅ **DiversityEnhancingAgent**: HILE和IRQN算法的详细实现
3. ✅ **使用示例**: 展示如何使用DiQoG API
4. ✅ **配置文件**: 完整的系统配置
5. ✅ **测试用例**: 单元测试示例

可以基于这些代码框架开始具体的开发工作!
