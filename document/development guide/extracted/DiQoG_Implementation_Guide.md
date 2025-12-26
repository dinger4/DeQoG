# DiQoG 代码项目实现思路与开发大纲

## 项目概述

DiQoG (Diversity-Driven Quality-Assured Code Generation) 是一个基于软件工程方法论的GenAI系统框架,用于自动化容错N版本代码生成。本文档提供完整的代码实现架构和开发指南。

---

## 一、总体架构设计

### 1.1 系统分层架构

```
┌─────────────────────────────────────────────┐
│         FSM Layer (状态机层)                  │
│  - State Controller                          │
│  - Transition Decision Engine                │
│  - Context Memory Manager                    │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│         LLM Agents Layer (LLM代理层)         │
│  - Understanding & Observing Agent           │
│  - Diversity Enhancing Agent                 │
│  - Code Generating Agent                     │
│  - Decision-making & Evaluating Agent        │
└─────────────────────────────────────────────┘
                    ↕
┌─────────────────────────────────────────────┐
│         Tools Layer (工具层)                  │
│  - Dynamic Prompt Generator                  │
│  - Knowledge Search                          │
│  - Diversity Evaluator                       │
│  - Code Interpreter                          │
│  - Test Cases Executor                       │
│  - Debugger                                  │
│  - Code Collector                            │
└─────────────────────────────────────────────┘
```

### 1.2 核心设计原则

1. **认知解耦**: LLM层专注于推理决策,工具层负责具体执行
2. **状态可控**: FSM确保系统行为的确定性和可预测性
3. **多样性驱动**: 从需求分析到实现全程注入多样性
4. **质量保证**: 基于反馈的迭代优化机制

---

## 二、核心模块实现

### 2.1 FSM Layer 实现

#### 2.1.1 状态定义

```python
from enum import Enum, auto

class SystemState(Enum):
    """系统五状态定义"""
    STATE_1_UNDERSTANDING = auto()  # 问题理解与信息收集
    STATE_2_DIVERSITY_IDEATION = auto()  # 多样性驱动的思路生成
    STATE_3_CODE_SYNTHESIS = auto()  # 代码实现综合
    STATE_4_QUALITY_VALIDATION = auto()  # 质量验证与调试
    STATE_5_COLLECTION = auto()  # N版本代码收集
    STATE_ERROR = auto()  # 错误状态
    STATE_COMPLETE = auto()  # 完成状态
```

#### 2.1.2 状态控制器设计

```python
class StateController:
    """
    状态控制器 - FSM层核心组件
    负责: 状态转换决策、上下文管理、rollback机制
    """
    
    def __init__(self, llm_client, config):
        self.current_state = SystemState.STATE_1_UNDERSTANDING
        self.llm_client = llm_client
        self.config = config
        self.context_memory = ContextMemory()
        self.transition_history = []
        
    def evaluate_transition(self, current_output, tools_result):
        """
        评估是否满足状态转换条件
        Returns: (next_state, action_type)
        action_type: 'transition', 'retry', 'rollback'
        """
        pass
    
    def execute_transition(self, next_state, carry_forward_data):
        """执行状态转换,携带必要的上下文"""
        pass
    
    def handle_retry(self, error_info):
        """处理可恢复的错误,更新prompt后重试"""
        pass
    
    def handle_rollback(self, target_state, failure_reason):
        """处理根本性缺陷,回滚到之前状态"""
        pass
```

#### 2.1.3 上下文记忆管理

```python
class ContextMemory:
    """
    跨状态的上下文记忆
    存储: 任务描述、生成历史、反馈信息、工具输出
    """
    
    def __init__(self):
        self.task_context = {}
        self.generation_history = []
        self.feedback_accumulation = []
        self.tool_outputs_cache = {}
        
    def update_context(self, state, data):
        """更新特定状态的上下文"""
        pass
    
    def get_state_context(self, state):
        """获取特定状态需要的上下文"""
        pass
    
    def persist_to_next_state(self, source_state, target_state):
        """在状态转换时持久化关键信息"""
        pass
```

---

### 2.2 LLM Agents Layer 实现

#### 2.2.1 基础Agent抽象类

```python
from abc import ABC, abstractmethod

class BaseLLMAgent(ABC):
    """
    LLM Agent基类
    定义所有Agent的通用接口和行为
    """
    
    def __init__(self, llm_client, role_prompt, available_tools):
        self.llm_client = llm_client
        self.role_prompt = role_prompt
        self.available_tools = available_tools
        
    @abstractmethod
    def process(self, input_data, context):
        """处理输入并生成输出"""
        pass
    
    def select_and_invoke_tool(self, tool_name, params):
        """自主选择并调用工具"""
        if tool_name in self.available_tools:
            return self.available_tools[tool_name].execute(params)
        return None
    
    def construct_prompt(self, input_data, context, state_info):
        """构建状态特定的提示"""
        pass
```

#### 2.2.2 专用Agent实现

```python
class UnderstandingAgent(BaseLLMAgent):
    """
    State 1: 问题理解与信息收集Agent
    工具: Knowledge Search, Dynamic Prompt Generator
    """
    
    def process(self, task_description, context):
        """
        1. 解析任务描述
        2. 调用Knowledge Search收集相关知识
        3. 生成问题理解报告
        """
        pass

class DiversityEnhancingAgent(BaseLLMAgent):
    """
    State 2: 多样性增强Agent
    实现HILE算法 (Hierarchical Isolation and Local Expansion)
    工具: Diversity Evaluator, Dynamic Prompt Generator
    """
    
    def process(self, understanding_result, context):
        """
        1. 生成多层次多样化思路
           - 算法层面 (自然语言)
           - 伪代码层面
           - 具体实现方案
        2. 使用Diversity Evaluator评估多样性
        3. 应用IRQN方法进一步增强多样性
        """
        pass
    
    def apply_HILE(self, task_info):
        """
        实现HILE算法:
        - 分层隔离: 思想层、方案层、实现层
        - 局部扩展: 在每一层进行深度探索
        """
        pass
    
    def apply_IRQN(self, initial_outputs, knowledge_base):
        """
        实现IRQN方法:
        Iterative Retention, Questioning and Negation
        - Retain: 保留完全不同的方案
        - Question: 质疑部分相似的方案
        - Negate: 否定并重新生成
        """
        pass

class CodeGeneratingAgent(BaseLLMAgent):
    """
    State 3: 代码生成Agent
    工具: Code Interpreter, Implementation Veto
    """
    
    def process(self, diverse_ideas, context):
        """
        1. 将每个多样化思路转换为可执行代码
        2. 使用Code Interpreter验证语法
        3. Implementation Veto机制拒绝非多样化变体
        """
        pass

class EvaluatingAgent(BaseLLMAgent):
    """
    State 4: 评估与调试Agent
    工具: Test Cases Executor, Debugger
    """
    
    def process(self, generated_codes, test_cases, context):
        """
        1. 执行测试用例
        2. 识别bug
        3. 整合测试失败反馈到refinement prompts
        4. 迭代修复
        """
        pass
    
    def iterative_refinement(self, code, test_failures):
        """基于测试反馈的迭代refinement"""
        pass
```

---

### 2.3 Tools Layer 实现

#### 2.3.1 工具接口定义

```python
class BaseTool(ABC):
    """工具基类"""
    
    @abstractmethod
    def execute(self, params):
        """执行工具功能"""
        pass
    
    @abstractmethod
    def validate_params(self, params):
        """验证参数"""
        pass
```

#### 2.3.2 核心工具实现

```python
class DynamicPromptGenerator(BaseTool):
    """
    动态提示生成器
    根据当前状态、历史上下文、执行反馈自适应构建提示
    """
    
    def __init__(self):
        self.prompt_templates = self.load_templates()
        
    def execute(self, params):
        """
        params: {
            'state': current_state,
            'task_info': task_description,
            'history': generation_history,
            'feedback': execution_feedback
        }
        """
        state = params['state']
        template = self.get_state_template(state)
        prompt = self.fill_template(template, params)
        return prompt
    
    def get_state_template(self, state):
        """获取状态特定的模板"""
        pass
    
    def integrate_feedback(self, template, feedback):
        """将执行反馈整合到提示中"""
        pass

class DiversityEvaluator(BaseTool):
    """
    多维度多样性评估器
    - 语义相似度 (CodeBERT embeddings)
    - 方法论差异 (LLM-based comparison)
    - 执行路径分歧
    """
    
    def execute(self, params):
        """
        params: {
            'code_list': [code1, code2, ...],
            'evaluation_type': 'semantic'|'methodological'|'execution'
        }
        Returns: diversity_score
        """
        pass
    
    def calculate_semantic_similarity(self, code1, code2):
        """使用CodeBERT计算语义相似度"""
        pass
    
    def calculate_methodological_difference(self, code1, code2):
        """使用LLM评估方法论差异"""
        pass
    
    def calculate_execution_path_divergence(self, code1, code2):
        """通过执行路径分析计算差异"""
        pass
    
    def compute_mbcs(self, code_list):
        """
        计算Mean BERT Cosine Similarity
        DiQoG论文中的核心指标之一
        """
        pass
    
    def compute_sdp(self, code_list):
        """
        计算Solutions Difference Probability
        衡量算法和方法论多样性
        """
        pass

class CodeInterpreter(BaseTool):
    """
    代码解释器
    在沙箱环境中执行代码,捕获运行时行为和错误信息
    """
    
    def execute(self, params):
        """
        params: {
            'code': code_string,
            'timeout': execution_timeout
        }
        Returns: {
            'success': bool,
            'output': execution_output,
            'error': error_message
        }
        """
        pass
    
    def validate_syntax(self, code):
        """语法验证"""
        pass
    
    def execute_in_sandbox(self, code):
        """沙箱执行"""
        pass

class TestCasesExecutor(BaseTool):
    """
    测试用例执行器
    运行comprehensive test suites,验证功能正确性
    """
    
    def execute(self, params):
        """
        params: {
            'code': code_string,
            'test_cases': [test1, test2, ...]
        }
        Returns: {
            'pass_rate': float,
            'failed_cases': [case_info, ...],
            'execution_details': details
        }
        """
        pass
    
    def run_single_test(self, code, test_case):
        """执行单个测试用例"""
        pass
    
    def aggregate_results(self, test_results):
        """聚合测试结果"""
        pass

class KnowledgeSearch(BaseTool):
    """
    知识检索工具
    从预设的知识库中检索相关信息
    Knowledge bases: K = {K_algo, K_impl, K_f-t}
    """
    
    def __init__(self):
        self.knowledge_bases = {
            'algorithmic': self.load_algorithmic_patterns(),
            'implementation': self.load_implementation_techniques(),
            'fault_tolerance': self.load_fault_tolerance_strategies()
        }
    
    def execute(self, params):
        """
        params: {
            'query': search_query,
            'knowledge_type': 'algorithmic'|'implementation'|'fault_tolerance'
        }
        Returns: relevant_knowledge
        """
        pass
    
    def search_algorithmic_patterns(self, query):
        """搜索算法模式"""
        pass
    
    def search_implementation_techniques(self, query):
        """搜索实现技术"""
        pass

class Debugger(BaseTool):
    """
    调试器
    识别和修复bugs,基于测试反馈
    """
    
    def execute(self, params):
        """
        params: {
            'code': buggy_code,
            'test_failures': failure_info,
            'error_messages': errors
        }
        Returns: {
            'bug_analysis': analysis,
            'suggested_fixes': fixes
        }
        """
        pass
    
    def analyze_failure(self, code, test_failure):
        """分析测试失败原因"""
        pass
    
    def generate_fix_suggestions(self, bug_analysis):
        """生成修复建议"""
        pass

class CodeCollector(BaseTool):
    """
    代码收集器
    收集所有通过验证的N个版本
    """
    
    def execute(self, params):
        """
        params: {
            'validated_codes': [code1, code2, ...],
            'metadata': version_metadata
        }
        Returns: n_version_code_set
        """
        pass
```

---

### 2.4 多样性增强算法实现

#### 2.4.1 HILE算法

```python
class HILEAlgorithm:
    """
    Hierarchical Isolation and Local Expansion
    分层隔离与局部扩展方法
    """
    
    def __init__(self, knowledge_bases):
        self.knowledge_bases = knowledge_bases
        self.levels = {
            'thought': 'L_Thought',
            'solution': 'L_Solution', 
            'implementation': 'L_Implementation'
        }
    
    def execute(self, task_info, n_versions):
        """
        执行HILE算法
        Returns: {
            'thought_level': [思路1, 思路2, ...],
            'solution_level': [方案1, 方案2, ...],
            'implementation_level': [实现1, 实现2, ...]
        }
        """
        results = {}
        
        # Level 1: Thought-level diversity
        results['thought_level'] = self.explore_thought_level(
            task_info, n_versions
        )
        
        # Level 2: Solution-level diversity
        results['solution_level'] = self.explore_solution_level(
            results['thought_level'], n_versions
        )
        
        # Level 3: Implementation-level diversity
        results['implementation_level'] = self.explore_implementation_level(
            results['solution_level'], n_versions
        )
        
        return results
    
    def explore_thought_level(self, task_info, n):
        """
        思想层探索
        生成不同的算法思路 (自然语言描述)
        """
        pass
    
    def explore_solution_level(self, thoughts, n):
        """
        方案层探索
        为每个思路生成伪代码级别的实现策略
        """
        pass
    
    def explore_implementation_level(self, solutions, n):
        """
        实现层探索
        为每个方案生成具体的代码实现方案
        包括: 数据结构选择、控制流模式、库函数调用
        """
        pass
```

#### 2.4.2 IRQN方法

```python
class IRQNMethod:
    """
    Iterative Retention, Questioning and Negation
    迭代保留、质疑与否定方法
    """
    
    def __init__(self, diversity_evaluator, llm_client):
        self.diversity_evaluator = diversity_evaluator
        self.llm_client = llm_client
        self.theta_diff = 0.3  # 完全不同的阈值
        self.theta_ident = 0.7  # 基本相同的阈值
    
    def execute(self, initial_outputs, knowledge_base, p_qn1, p_qn2, max_iterations):
        """
        执行IRQN方法
        
        Parameters:
        - initial_outputs: 初始输出集合
        - knowledge_base: 领域知识库
        - p_qn1: 触发深度判断的概率
        - p_qn2: 对保留元素进行否定的概率
        - max_iterations: 最大迭代次数
        
        Returns: 最终输出集合
        """
        final_outputs = []
        pending_outputs = initial_outputs.copy()
        history = []
        
        for iteration in range(max_iterations):
            current_batch = []
            
            for output in pending_outputs:
                # 概率触发判断
                if random.random() > p_qn1:
                    # 直接接受
                    final_outputs.append(output)
                    continue
                
                # 评估相似度
                similarity = self.evaluate_similarity(
                    output, history + final_outputs
                )
                
                # 根据相似度决定操作
                action_result = self.decide_action(
                    output, similarity, p_qn2, knowledge_base
                )
                
                if action_result['action'] == 'retain':
                    final_outputs.append(action_result['output'])
                elif action_result['action'] == 'question':
                    current_batch.append(action_result['output'])
                elif action_result['action'] == 'negate':
                    current_batch.append(action_result['output'])
                
            pending_outputs = current_batch
            history.extend(final_outputs)
            
            if not pending_outputs:
                break
        
        return final_outputs
    
    def evaluate_similarity(self, output, reference_set):
        """评估输出与参考集的相似度"""
        if not reference_set:
            return 0.0
        
        similarities = [
            self.diversity_evaluator.calculate_semantic_similarity(
                output, ref
            )
            for ref in reference_set
        ]
        return max(similarities)
    
    def decide_action(self, output, similarity, p_qn2, knowledge_base):
        """
        根据相似度决定采取的行动
        - Retain: similarity < theta_diff
        - Question: theta_diff <= similarity <= theta_ident
        - Negate: similarity > theta_ident
        """
        if similarity < self.theta_diff:
            # Retain: 完全不同,保留
            if random.random() < p_qn2:
                # 进一步否定以产生更多多样性
                negated = self.negate_and_regenerate(output, knowledge_base)
                return {'action': 'negate', 'output': negated}
            return {'action': 'retain', 'output': output}
        
        elif similarity <= self.theta_ident:
            # Question: 部分相似,质疑并重新生成
            questioned = self.question_and_refine(output, knowledge_base)
            return {'action': 'question', 'output': questioned}
        
        else:
            # Negate: 过于相似,否定并重新生成
            negated = self.negate_and_regenerate(output, knowledge_base)
            return {'action': 'negate', 'output': negated}
    
    def question_and_refine(self, output, knowledge_base):
        """质疑并细化输出"""
        prompt = f"""
        Current output: {output}
        
        This output shows partial similarity with existing solutions.
        Please refine it by:
        1. Identifying what makes it similar
        2. Proposing alternative approaches
        3. Enhancing its distinctiveness
        
        Knowledge base: {knowledge_base}
        """
        refined = self.llm_client.generate(prompt)
        return refined
    
    def negate_and_regenerate(self, output, knowledge_base):
        """否定当前输出并重新生成"""
        prompt = f"""
        Current output: {output}
        
        This approach has been used. Please generate a completely different solution that:
        1. Uses a different algorithmic paradigm
        2. Employs different data structures
        3. Takes a contrasting approach to solve the problem
        
        Knowledge base: {knowledge_base}
        """
        regenerated = self.llm_client.generate(prompt)
        return regenerated
```

---

### 2.5 质量保证机制实现

```python
class QualityAssuranceEngine:
    """
    质量保证引擎
    实现基于反馈的迭代优化
    """
    
    def __init__(self, test_executor, debugger, code_interpreter):
        self.test_executor = test_executor
        self.debugger = debugger
        self.code_interpreter = code_interpreter
        self.max_refinement_iterations = 5
    
    def validate_and_refine(self, code, test_cases, context):
        """
        验证并迭代改进代码
        Returns: {
            'code': refined_code,
            'pass_rate': float,
            'iterations': int,
            'refinement_history': [...]
        }
        """
        current_code = code
        refinement_history = []
        
        for iteration in range(self.max_refinement_iterations):
            # 执行测试
            test_result = self.test_executor.execute({
                'code': current_code,
                'test_cases': test_cases
            })
            
            # 记录迭代信息
            refinement_history.append({
                'iteration': iteration,
                'code': current_code,
                'test_result': test_result
            })
            
            # 如果通过所有测试,返回
            if test_result['pass_rate'] >= 1.0:
                return {
                    'code': current_code,
                    'pass_rate': test_result['pass_rate'],
                    'iterations': iteration + 1,
                    'refinement_history': refinement_history
                }
            
            # 分析失败并生成修复
            debug_result = self.debugger.execute({
                'code': current_code,
                'test_failures': test_result['failed_cases'],
                'error_messages': test_result.get('errors', [])
            })
            
            # 应用修复建议
            current_code = self.apply_fixes(
                current_code, 
                debug_result['suggested_fixes']
            )
        
        # 达到最大迭代次数
        return {
            'code': current_code,
            'pass_rate': test_result['pass_rate'],
            'iterations': self.max_refinement_iterations,
            'refinement_history': refinement_history,
            'warning': 'Max iterations reached'
        }
    
    def apply_fixes(self, code, fixes):
        """应用修复建议到代码"""
        pass
```

---

## 三、核心流程实现

### 3.1 主工作流程

```python
class DiQoGPipeline:
    """
    DiQoG主工作流程
    协调五个状态的执行
    """
    
    def __init__(self, config):
        self.config = config
        self.state_controller = StateController(
            llm_client=self.init_llm_client(),
            config=config
        )
        self.agents = self.init_agents()
        self.tools = self.init_tools()
        
    def generate_n_versions(self, task_description, test_cases, n=5):
        """
        生成N个版本的容错代码
        
        Parameters:
        - task_description: 任务描述
        - test_cases: 测试用例集
        - n: 版本数量
        
        Returns: {
            'n_version_codes': [code1, code2, ...],
            'diversity_metrics': {...},
            'quality_metrics': {...},
            'generation_metadata': {...}
        }
        """
        # State 1: Understanding
        understanding_result = self.execute_state_1(task_description)
        
        # State 2: Diversity Ideation
        diverse_ideas = self.execute_state_2(understanding_result, n)
        
        # State 3: Code Synthesis
        generated_codes = self.execute_state_3(diverse_ideas)
        
        # State 4: Quality Validation
        validated_codes = self.execute_state_4(generated_codes, test_cases)
        
        # State 5: Collection
        final_result = self.execute_state_5(validated_codes)
        
        return final_result
    
    def execute_state_1(self, task_description):
        """
        State 1: 问题理解与信息收集
        """
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_1_UNDERSTANDING
        )
        
        agent = self.agents['understanding']
        result = agent.process(task_description, context)
        
        # 评估是否满足转换条件
        transition = self.state_controller.evaluate_transition(
            result, None
        )
        
        if transition[1] == 'rollback':
            # 如果理解不充分,可能需要更多信息
            result = self.handle_understanding_insufficiency(result)
        
        self.state_controller.execute_transition(
            SystemState.STATE_2_DIVERSITY_IDEATION,
            result
        )
        
        return result
    
    def execute_state_2(self, understanding_result, n):
        """
        State 2: 多样性驱动的思路生成
        应用HILE算法
        """
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_2_DIVERSITY_IDEATION
        )
        
        agent = self.agents['diversity_enhancing']
        diverse_ideas = agent.process(understanding_result, context)
        
        # 评估多样性
        diversity_score = self.tools['diversity_evaluator'].compute_sdp(
            diverse_ideas
        )
        
        # 判断是否需要更多多样性
        while diversity_score < self.config.diversity_threshold:
            # 应用IRQN方法增强多样性
            diverse_ideas = agent.apply_IRQN(
                diverse_ideas,
                self.tools['knowledge_search']
            )
            diversity_score = self.tools['diversity_evaluator'].compute_sdp(
                diverse_ideas
            )
        
        self.state_controller.execute_transition(
            SystemState.STATE_3_CODE_SYNTHESIS,
            diverse_ideas
        )
        
        return diverse_ideas
    
    def execute_state_3(self, diverse_ideas):
        """
        State 3: 代码实现综合
        将思路转化为可执行代码
        """
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_3_CODE_SYNTHESIS
        )
        
        agent = self.agents['code_generating']
        generated_codes = []
        
        for idea in diverse_ideas:
            code = agent.process(idea, context)
            
            # 语法验证
            validation_result = self.tools['code_interpreter'].validate_syntax(code)
            
            if validation_result['success']:
                # Implementation Veto: 检查是否与已生成代码过于相似
                if self.is_sufficiently_diverse(code, generated_codes):
                    generated_codes.append(code)
                else:
                    # Rollback到State 2,请求新的idea
                    self.state_controller.handle_rollback(
                        SystemState.STATE_2_DIVERSITY_IDEATION,
                        "Generated code not diverse enough"
                    )
            else:
                # Retry with error feedback
                self.state_controller.handle_retry(validation_result['error'])
        
        self.state_controller.execute_transition(
            SystemState.STATE_4_QUALITY_VALIDATION,
            generated_codes
        )
        
        return generated_codes
    
    def execute_state_4(self, generated_codes, test_cases):
        """
        State 4: 质量验证与调试
        """
        context = self.state_controller.context_memory.get_state_context(
            SystemState.STATE_4_QUALITY_VALIDATION
        )
        
        agent = self.agents['evaluating']
        qa_engine = QualityAssuranceEngine(
            self.tools['test_executor'],
            self.tools['debugger'],
            self.tools['code_interpreter']
        )
        
        validated_codes = []
        
        for code in generated_codes:
            refinement_result = qa_engine.validate_and_refine(
                code, test_cases, context
            )
            
            if refinement_result['pass_rate'] >= self.config.quality_threshold:
                validated_codes.append({
                    'code': refinement_result['code'],
                    'metrics': {
                        'pass_rate': refinement_result['pass_rate'],
                        'iterations': refinement_result['iterations']
                    }
                })
        
        self.state_controller.execute_transition(
            SystemState.STATE_5_COLLECTION,
            validated_codes
        )
        
        return validated_codes
    
    def execute_state_5(self, validated_codes):
        """
        State 5: N版本代码收集
        """
        code_collector = self.tools['code_collector']
        
        result = code_collector.execute({
            'validated_codes': validated_codes,
            'metadata': self.state_controller.context_memory.task_context
        })
        
        self.state_controller.execute_transition(
            SystemState.STATE_COMPLETE,
            result
        )
        
        return result
    
    def is_sufficiently_diverse(self, new_code, existing_codes):
        """判断新代码是否与已有代码足够多样化"""
        if not existing_codes:
            return True
        
        similarities = [
            self.tools['diversity_evaluator'].calculate_semantic_similarity(
                new_code, existing
            )
            for existing in existing_codes
        ]
        
        return max(similarities) < self.config.diversity_threshold
```

---

## 四、评估指标实现

### 4.1 多样性指标

```python
class DiversityMetrics:
    """
    多样性评估指标
    实现MBCS和SDP
    """
    
    def __init__(self, codebert_model):
        self.codebert_model = codebert_model
    
    def compute_mbcs(self, code_list):
        """
        Mean BERT Cosine Similarity
        使用CodeBERT embeddings计算代码对之间的语义相似度
        """
        embeddings = [
            self.codebert_model.encode(code)
            for code in code_list
        ]
        
        n = len(embeddings)
        if n < 2:
            return 0.0
        
        total_similarity = 0
        pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                similarity = cosine_similarity(
                    embeddings[i], 
                    embeddings[j]
                )
                total_similarity += similarity
                pairs += 1
        
        mbcs = total_similarity / pairs if pairs > 0 else 0
        return mbcs
    
    def compute_sdp(self, code_list, llm_client):
        """
        Solutions Difference Probability
        使用LLM评估方法论和算法差异
        """
        n = len(code_list)
        if n < 2:
            return 0.0
        
        different_pairs = 0
        total_pairs = 0
        
        for i in range(n):
            for j in range(i+1, n):
                prompt = f"""
                Compare these two code implementations:
                
                Code 1:
                {code_list[i]}
                
                Code 2:
                {code_list[j]}
                
                Are they using different algorithmic approaches or methodologies?
                Answer with only 'Yes' or 'No'.
                """
                
                response = llm_client.generate(prompt)
                if 'yes' in response.lower():
                    different_pairs += 1
                total_pairs += 1
        
        sdp = different_pairs / total_pairs if total_pairs > 0 else 0
        return sdp
```

### 4.2 正确性指标

```python
class CorrectnessMetrics:
    """
    代码正确性评估指标
    """
    
    def compute_pass_rate(self, test_results):
        """
        计算测试通过率
        """
        total_tests = len(test_results)
        passed_tests = sum(1 for result in test_results if result['passed'])
        return passed_tests / total_tests if total_tests > 0 else 0
    
    def compute_tpr(self, test_results_per_version):
        """
        Test Pass Rate (TPR)
        所有版本的平均测试通过率
        """
        pass_rates = [
            self.compute_pass_rate(result)
            for result in test_results_per_version
        ]
        return sum(pass_rates) / len(pass_rates) if pass_rates else 0
```

### 4.3 容错能力指标

```python
class FaultToleranceMetrics:
    """
    容错能力评估指标
    用于fault injection实验
    """
    
    def compute_failure_rate(self, voting_results):
        """
        Failure Rate (FR)
        系统级别的失败率
        """
        total_cases = len(voting_results)
        failed_cases = sum(
            1 for result in voting_results 
            if not result['system_correct']
        )
        return failed_cases / total_cases if total_cases > 0 else 0
    
    def compute_mcr(self, voting_results):
        """
        Majority Consensus Rate (MCR)
        多数版本达成一致的比率
        """
        total_cases = len(voting_results)
        majority_consensus = sum(
            1 for result in voting_results 
            if result['has_majority_consensus']
        )
        return majority_consensus / total_cases if total_cases > 0 else 0
    
    def compute_ccr(self, voting_results):
        """
        Complete Consensus Rate (CCR)
        所有版本完全一致的比率
        """
        total_cases = len(voting_results)
        complete_consensus = sum(
            1 for result in voting_results 
            if result['all_agree']
        )
        return complete_consensus / total_cases if total_cases > 0 else 0
```

---

## 五、实验框架实现

### 5.1 Fault Injection实验

```python
class FaultInjectionExperiment:
    """
    故障注入实验框架
    实现代码级和算法级故障注入
    """
    
    def __init__(self, n_versions):
        self.n_versions = n_versions
        self.fault_patterns = self.define_fault_patterns()
    
    def define_fault_patterns(self):
        """
        定义故障注入模式
        """
        return {
            'code_level': [
                'Pat-CL 0',  # 无故障
                'Pat-CL 1',  # 恰好一个版本包含故障
                'Pat-CL 2',  # ⌊(N-1)/2⌋个版本包含故障
                'Pat-CL 3',  # ⌊(N+1)/2⌋个版本包含故障
                'Pat-CL 4'   # 所有版本都包含故障
            ],
            'algorithm_level': [
                'Pat-AL 0',  # 无CMF
                'Pat-AL 1',  # 所有版本包含恰好一个CMF
                'Pat-AL 2',  # 所有版本包含⌊(N-1)/2⌋个CMF
                'Pat-AL 3',  # 所有版本包含⌊(N+1)/2⌋个CMF
                'Pat-AL 4'   # 所有版本包含所有可用CMF
            ]
        }
    
    def inject_code_level_faults(self, codes, pattern):
        """
        注入代码级故障
        - 语法错误
        - 语义错误
        - 逻辑错误
        """
        n = len(codes)
        faulty_codes = codes.copy()
        
        if pattern == 'Pat-CL 1':
            fault_indices = [0]
        elif pattern == 'Pat-CL 2':
            num_faults = (n - 1) // 2
            fault_indices = random.sample(range(n), num_faults)
        elif pattern == 'Pat-CL 3':
            num_faults = (n + 1) // 2
            fault_indices = random.sample(range(n), num_faults)
        elif pattern == 'Pat-CL 4':
            fault_indices = range(n)
        else:  # Pat-CL 0
            fault_indices = []
        
        for idx in fault_indices:
            faulty_codes[idx] = self.introduce_code_fault(codes[idx])
        
        return faulty_codes
    
    def inject_algorithm_level_faults(self, codes, pattern, cmf_library):
        """
        注入算法级故障 (Common Mode Failures)
        - 错误的算法选择
        - 共同的逻辑缺陷
        """
        n = len(codes)
        faulty_codes = []
        
        if pattern == 'Pat-AL 1':
            cmf = random.choice(cmf_library)
            faulty_codes = [
                self.apply_cmf(code, cmf) 
                for code in codes
            ]
        elif pattern == 'Pat-AL 2':
            num_cmfs = (n - 1) // 2
            selected_cmfs = random.sample(cmf_library, num_cmfs)
            faulty_codes = [
                self.apply_multiple_cmfs(code, selected_cmfs)
                for code in codes
            ]
        # ... 其他模式类似
        
        return faulty_codes
    
    def run_experiment(self, n_version_codes, test_cases, patterns):
        """
        运行完整的故障注入实验
        """
        results = {
            'code_level': {},
            'algorithm_level': {}
        }
        
        # 代码级故障注入
        for pattern in patterns['code_level']:
            faulty_codes = self.inject_code_level_faults(
                n_version_codes, pattern
            )
            voting_result = self.majority_voting(faulty_codes, test_cases)
            results['code_level'][pattern] = self.evaluate_result(voting_result)
        
        # 算法级故障注入
        for pattern in patterns['algorithm_level']:
            faulty_codes = self.inject_algorithm_level_faults(
                n_version_codes, pattern, self.cmf_library
            )
            voting_result = self.majority_voting(faulty_codes, test_cases)
            results['algorithm_level'][pattern] = self.evaluate_result(voting_result)
        
        return results
    
    def majority_voting(self, codes, test_cases):
        """
        多数投票机制
        """
        voting_results = []
        
        for test_case in test_cases:
            outputs = []
            for code in codes:
                try:
                    output = execute_code(code, test_case['input'])
                    outputs.append(output)
                except Exception as e:
                    outputs.append(None)
            
            # 投票选择最终输出
            final_output = self.vote(outputs)
            correct = (final_output == test_case['expected_output'])
            
            voting_results.append({
                'outputs': outputs,
                'final_output': final_output,
                'system_correct': correct,
                'has_majority_consensus': self.check_majority(outputs),
                'all_agree': self.check_complete_consensus(outputs)
            })
        
        return voting_results
```

---

## 六、项目结构

```
DiQoG/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── fsm_controller.py          # FSM层核心控制器
│   │   ├── context_memory.py          # 上下文记忆管理
│   │   └── pipeline.py                # 主工作流程
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py              # Agent基类
│   │   ├── understanding_agent.py     # State 1 Agent
│   │   ├── diversity_agent.py         # State 2 Agent
│   │   ├── code_generating_agent.py   # State 3 Agent
│   │   └── evaluating_agent.py        # State 4 Agent
│   │
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base_tool.py               # 工具基类
│   │   ├── prompt_generator.py        # 动态提示生成器
│   │   ├── diversity_evaluator.py     # 多样性评估器
│   │   ├── code_interpreter.py        # 代码解释器
│   │   ├── test_executor.py           # 测试执行器
│   │   ├── debugger.py                # 调试器
│   │   └── knowledge_search.py        # 知识检索
│   │
│   ├── algorithms/
│   │   ├── __init__.py
│   │   ├── hile.py                    # HILE算法实现
│   │   ├── irqn.py                    # IRQN方法实现
│   │   └── quality_assurance.py       # 质量保证引擎
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── diversity_metrics.py       # 多样性指标
│   │   ├── correctness_metrics.py     # 正确性指标
│   │   └── fault_tolerance_metrics.py # 容错指标
│   │
│   ├── experiments/
│   │   ├── __init__.py
│   │   ├── fault_injection.py         # 故障注入实验
│   │   └── ablation_study.py          # 消融实验
│   │
│   └── utils/
│       ├── __init__.py
│       ├── llm_client.py              # LLM客户端封装
│       ├── config.py                  # 配置管理
│       └── logger.py                  # 日志管理
│
├── data/
│   ├── knowledge_bases/               # 知识库
│   │   ├── algorithmic_patterns.json
│   │   ├── implementation_techniques.json
│   │   └── fault_tolerance_strategies.json
│   │
│   ├── datasets/                      # 数据集
│   │   ├── MBPP/
│   │   ├── HumanEval/
│   │   ├── ClassEval/
│   │   └── MIPD/
│   │
│   └── prompts/                       # 提示模板
│       ├── state_1_templates.json
│       ├── state_2_templates.json
│       ├── state_3_templates.json
│       └── state_4_templates.json
│
├── experiments/                        # 实验脚本
│   ├── run_rq1_diversity.py
│   ├── run_rq2_fault_tolerance.py
│   ├── run_rq3_cost_analysis.py
│   └── run_rq4_ablation.py
│
├── tests/                             # 单元测试
│   ├── test_fsm.py
│   ├── test_agents.py
│   ├── test_tools.py
│   └── test_algorithms.py
│
├── configs/                           # 配置文件
│   ├── default_config.yaml
│   └── experiment_configs/
│
├── notebooks/                         # Jupyter notebooks (分析&可视化)
│   ├── results_analysis.ipynb
│   └── visualization.ipynb
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## 七、开发优先级与里程碑

### Phase 1: 核心框架搭建 (Week 1-2)

**目标**: 建立基本的FSM框架和工具层

1. ✅ 实现`SystemState`枚举和`StateController`
2. ✅ 实现`BaseTool`和基础工具
3. ✅ 实现`BaseLLMAgent`和LLM客户端封装
4. ✅ 实现`ContextMemory`上下文管理

**验证**: 能够运行一个简单的单状态转换

### Phase 2: 多样性增强实现 (Week 3-4)

**目标**: 实现HILE和IRQN算法

1. ✅ 实现`HILEAlgorithm`三层探索
2. ✅ 实现`IRQNMethod`迭代优化
3. ✅ 实现`DiversityEvaluator`工具
4. ✅ 集成到`DiversityEnhancingAgent`

**验证**: 能够生成多样化的代码思路

### Phase 3: 完整流程实现 (Week 5-6)

**目标**: 打通五状态完整流程

1. ✅ 实现所有状态的Agent
2. ✅ 实现状态转换逻辑(transition, retry, rollback)
3. ✅ 实现`DiQoGPipeline`主流程
4. ✅ 集成质量保证引擎

**验证**: 能够端到端生成N版本代码

### Phase 4: 评估与实验 (Week 7-8)

**目标**: 实现评估指标和实验框架

1. ✅ 实现所有评估指标(MBCS, SDP, TPR, FR, MCR, CCR)
2. ✅ 实现故障注入实验框架
3. ✅ 在benchmark数据集上运行实验
4. ✅ 收集和分析实验结果

**验证**: 复现论文中的实验结果

### Phase 5: 优化与完善 (Week 9-10)

**目标**: 性能优化和文档完善

1. ✅ 性能分析和优化
2. ✅ 完善单元测试
3. ✅ 编写API文档
4. ✅ 准备开源发布

---

## 八、关键技术实现细节

### 8.1 LLM客户端封装

```python
class LLMClient:
    """
    LLM客户端统一接口
    支持多种LLM后端: GPT-4, Claude, DeepSeek等
    """
    
    def __init__(self, model_name, api_key, config):
        self.model_name = model_name
        self.api_key = api_key
        self.config = config
        self.client = self.initialize_client()
    
    def generate(self, prompt, temperature=0.7, max_tokens=2000):
        """
        生成文本
        支持temperature和max_tokens控制
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are an expert programmer."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_with_tools(self, prompt, tools, temperature=0.7):
        """
        支持工具调用的生成
        实现LLM主动调用工具的能力
        """
        pass
```

### 8.2 提示模板管理

```python
class PromptTemplateManager:
    """
    提示模板管理器
    加载和管理不同状态的提示模板
    """
    
    def __init__(self, template_dir):
        self.templates = self.load_templates(template_dir)
    
    def load_templates(self, template_dir):
        """从JSON文件加载模板"""
        templates = {}
        for state in SystemState:
            template_file = f"{template_dir}/state_{state.value}_templates.json"
            with open(template_file, 'r') as f:
                templates[state] = json.load(f)
        return templates
    
    def get_template(self, state, template_type):
        """获取特定状态和类型的模板"""
        return self.templates[state][template_type]
    
    def fill_template(self, template, variables):
        """填充模板变量"""
        return template.format(**variables)
```

### 8.3 代码执行沙箱

```python
class CodeSandbox:
    """
    安全的代码执行沙箱
    隔离执行环境,防止恶意代码
    """
    
    def __init__(self, timeout=5):
        self.timeout = timeout
    
    def execute(self, code, test_input):
        """
        在沙箱中执行代码
        使用subprocess + timeout控制
        """
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # 执行代码
            result = subprocess.run(
                ['python', temp_file],
                input=str(test_input),
                capture_output=True,
                timeout=self.timeout,
                text=True
            )
            
            if result.returncode == 0:
                return {
                    'success': True,
                    'output': result.stdout,
                    'error': None
                }
            else:
                return {
                    'success': False,
                    'output': None,
                    'error': result.stderr
                }
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'output': None,
                'error': 'Execution timeout'
            }
        finally:
            os.unlink(temp_file)
```

---

## 九、配置文件示例

```yaml
# default_config.yaml

# LLM配置
llm:
  model_name: "gpt-4"
  api_key: "${OPENAI_API_KEY}"
  temperature: 0.7
  max_tokens: 2000

# 多样性配置
diversity:
  threshold: 0.6  # 多样性阈值
  hile:
    num_thoughts: 5  # 思想层探索数量
    num_solutions: 3  # 方案层探索数量
    num_implementations: 2  # 实现层探索数量
  irqn:
    p_qn1: 0.7  # 触发判断概率
    p_qn2: 0.3  # 否定概率
    max_iterations: 5
    theta_diff: 0.3
    theta_ident: 0.7

# 质量保证配置
quality:
  threshold: 0.9  # 质量阈值
  max_refinement_iterations: 5
  test_timeout: 5  # 测试超时时间(秒)

# FSM配置
fsm:
  max_retries: 3  # 最大重试次数
  enable_rollback: true

# N版本配置
n_versions:
  default_n: 5
  min_n: 3
  max_n: 10

# 实验配置
experiments:
  datasets:
    - "MBPP"
    - "HumanEval"
    - "ClassEval"
    - "MIPD"
  fault_injection:
    code_level_patterns: ["Pat-CL 0", "Pat-CL 1", "Pat-CL 2", "Pat-CL 3", "Pat-CL 4"]
    algorithm_level_patterns: ["Pat-AL 0", "Pat-AL 1", "Pat-AL 2", "Pat-AL 3", "Pat-AL 4"]

# 日志配置
logging:
  level: "INFO"
  file: "logs/diqog.log"
```

---

## 十、使用示例

```python
# example_usage.py

from diqog import DiQoGPipeline, Config

# 1. 加载配置
config = Config.from_yaml('configs/default_config.yaml')

# 2. 初始化DiQoG pipeline
pipeline = DiQoGPipeline(config)

# 3. 定义任务
task_description = """
Write a Python function that calculates the nth Fibonacci number.
The function should handle edge cases and be efficient.

Function signature:
def fibonacci(n: int) -> int:
    pass
"""

test_cases = [
    {'input': 0, 'expected_output': 0},
    {'input': 1, 'expected_output': 1},
    {'input': 5, 'expected_output': 5},
    {'input': 10, 'expected_output': 55},
]

# 4. 生成N版本代码
result = pipeline.generate_n_versions(
    task_description=task_description,
    test_cases=test_cases,
    n=5
)

# 5. 查看结果
print(f"Generated {len(result['n_version_codes'])} diverse implementations")
print(f"Diversity metrics: {result['diversity_metrics']}")
print(f"Quality metrics: {result['quality_metrics']}")

# 6. 运行故障注入实验
from diqog.experiments import FaultInjectionExperiment

fi_experiment = FaultInjectionExperiment(n_versions=5)
fi_results = fi_experiment.run_experiment(
    n_version_codes=result['n_version_codes'],
    test_cases=test_cases,
    patterns={
        'code_level': ['Pat-CL 0', 'Pat-CL 1', 'Pat-CL 3'],
        'algorithm_level': ['Pat-AL 0', 'Pat-AL 1']
    }
)

print(f"Fault injection results: {fi_results}")
```

---

## 十一、测试策略

### 11.1 单元测试

```python
# tests/test_fsm.py

import unittest
from diqog.core import StateController, SystemState

class TestStateController(unittest.TestCase):
    
    def setUp(self):
        self.controller = StateController(llm_client=MockLLMClient(), config=test_config)
    
    def test_state_transition(self):
        """测试状态转换"""
        self.assertEqual(self.controller.current_state, SystemState.STATE_1_UNDERSTANDING)
        
        self.controller.execute_transition(SystemState.STATE_2_DIVERSITY_IDEATION, {})
        self.assertEqual(self.controller.current_state, SystemState.STATE_2_DIVERSITY_IDEATION)
    
    def test_rollback(self):
        """测试回滚机制"""
        self.controller.execute_transition(SystemState.STATE_3_CODE_SYNTHESIS, {})
        self.controller.handle_rollback(SystemState.STATE_2_DIVERSITY_IDEATION, "Insufficient diversity")
        
        self.assertEqual(self.controller.current_state, SystemState.STATE_2_DIVERSITY_IDEATION)
    
    def test_retry(self):
        """测试重试机制"""
        # ... 测试retry逻辑
        pass
```

### 11.2 集成测试

```python
# tests/test_integration.py

import unittest
from diqog import DiQoGPipeline

class TestDiQoGIntegration(unittest.TestCase):
    
    def setUp(self):
        self.pipeline = DiQoGPipeline(test_config)
    
    def test_end_to_end_generation(self):
        """测试端到端代码生成"""
        task = simple_task_description
        test_cases = simple_test_cases
        
        result = self.pipeline.generate_n_versions(task, test_cases, n=3)
        
        self.assertEqual(len(result['n_version_codes']), 3)
        self.assertGreater(result['diversity_metrics']['mbcs'], 0)
        self.assertGreater(result['quality_metrics']['tpr'], 0.8)
```

---

## 十二、性能优化建议

### 12.1 LLM调用优化

1. **批量处理**: 合并多个小请求为一个大请求
2. **缓存机制**: 缓存相似prompt的结果
3. **并行生成**: 多个版本并行生成而非串行

```python
class OptimizedLLMClient(LLMClient):
    """
    优化的LLM客户端
    支持批量处理和缓存
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
    
    def generate_batch(self, prompts, temperature=0.7):
        """批量生成,减少API调用次数"""
        results = []
        for prompt in prompts:
            # 检查缓存
            cache_key = self.get_cache_key(prompt, temperature)
            if cache_key in self.cache:
                results.append(self.cache[cache_key])
            else:
                result = self.generate(prompt, temperature)
                self.cache[cache_key] = result
                results.append(result)
        return results
```

### 12.2 多进程/多线程

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

class ParallelCodeGenerator:
    """
    并行代码生成器
    使用多线程/多进程加速
    """
    
    def generate_versions_parallel(self, ideas, n_workers=4):
        """并行生成多个版本"""
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self.generate_single_version, idea)
                for idea in ideas
            ]
            results = [future.result() for future in futures]
        return results
```

---

## 总结

本文档提供了DiQoG项目的完整实现思路,涵盖:

1. ✅ **系统架构**: 三层架构设计(FSM层、LLM层、工具层)
2. ✅ **核心算法**: HILE和IRQN的详细实现
3. ✅ **工作流程**: 五状态完整流程设计
4. ✅ **评估指标**: 多样性、正确性、容错能力指标
5. ✅ **实验框架**: 故障注入和消融实验
6. ✅ **项目结构**: 清晰的模块组织
7. ✅ **开发计划**: 分阶段的开发里程碑

按照此大纲进行开发,可以系统化地实现DiQoG框架,并在benchmark数据集上验证其性能,为论文投稿提供坚实的实验基础。

---

**下一步行动**:
1. 搭建项目骨架,创建基本的目录结构
2. 实现FSM核心控制器
3. 实现基础工具和Agent
4. 逐步完成HILE和IRQN算法
5. 在小规模数据集上测试
6. 扩展到完整benchmark并收集实验数据
