#!/usr/bin/env python3

"""
测试 tool_version 参数的功能
"""

def test_tool_version_parameter():
    """测试 VideoRLQwen 的 tool_version 参数"""
    
    # 测试默认值
    print("测试 1: 默认 tool_version")
    try:
        from vlmeval.vlm.video_tool_rl_gao.video_tool_rl import VideoRLQwen
        
        # 仅测试参数传递，不实际初始化模型
        import inspect
        sig = inspect.signature(VideoRLQwen.__init__)
        params = sig.parameters
        
        # 检查是否有 tool_version 参数
        if 'tool_version' in params:
            print("✅ tool_version 参数已添加到 VideoRLQwen.__init__")
            default_value = params['tool_version'].default
            print(f"✅ 默认值: {default_value}")
        else:
            print("❌ tool_version 参数未找到")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    # 测试 agent_rollout_loop 函数签名
    print("\n测试 2: agent_rollout_loop 函数签名")
    try:
        from vlmeval.vlm.video_tool_rl_gao.video_tools.inference_loop import agent_rollout_loop
        
        sig = inspect.signature(agent_rollout_loop)
        params = sig.parameters
        
        if 'tool_version' in params:
            print("✅ tool_version 参数已添加到 agent_rollout_loop")
            default_value = params['tool_version'].default
            print(f"✅ 默认值: {default_value}")
        else:
            print("❌ tool_version 参数未找到在 agent_rollout_loop")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

    print("\n测试 3: 检查代码修改")
    # 检查 parallel_envs.py 中的修改
    try:
        with open('/workspace/codes/VLMEvalKit/vlmeval/vlm/video_tool_rl_gao/video_tools/parallel_envs.py', 'r') as f:
            content = f.read()
            
        if "tool_version = kwargs.get('tool_version', 'video_toolbox')" in content:
            print("✅ parallel_envs.py 中的 tool_version 获取逻辑已添加")
        else:
            print("❌ parallel_envs.py 中未找到 tool_version 获取逻辑")
            
        if "tool_name = tool_version" in content:
            print("✅ parallel_envs.py 中的 tool_name 赋值逻辑已修改")
        else:
            print("❌ parallel_envs.py 中的 tool_name 赋值逻辑未修改")
            
    except Exception as e:
        print(f"❌ 检查文件失败: {e}")

if __name__ == "__main__":
    test_tool_version_parameter()
