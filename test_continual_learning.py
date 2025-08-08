#!/usr/bin/env python3
"""
연속 학습 테스트 스크립트
Usage: python test_continual_learning.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dreamerv3.task_manager import TaskManager


def test_task_manager():
    """TaskManager 기본 기능 테스트"""
    print("=== TaskManager 기본 테스트 ===")
    
    # Sequential 전략 테스트
    manager = TaskManager(
        domain="dmc_cartpole",
        tasks=["swingup", "balance", "swingup_sparse"],
        strategy="sequential",
        switch_interval=3
    )
    
    print(f"초기 태스크: {manager.get_current_task_name()}")
    
    # 에피소드 시뮬레이션
    for episode in range(10):
        manager.on_episode_start()
        
        if manager.should_switch_task():
            old_task = manager.get_current_task()
            new_task = manager.switch_task()
            print(f"Episode {episode}: 태스크 변경 {old_task} -> {new_task}")
        
        manager.on_episode_end(episode_reward=100, episode_length=200)
        print(f"Episode {episode}: 현재 태스크 = {manager.get_current_task_name()}")
    
    print("\n=== 최종 통계 ===")
    stats = manager.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")


def test_random_strategy():
    """Random 전략 테스트"""
    print("\n=== Random 전략 테스트 ===")
    
    manager = TaskManager(
        domain="dmc_walker",
        tasks=["stand", "walk", "run"],
        strategy="random",
        switch_interval=2,
        random_seed=42
    )
    
    for episode in range(8):
        if manager.should_switch_task():
            manager.switch_task()
        
        print(f"Episode {episode}: {manager.get_current_task_name()}")
        manager.on_episode_end()


def test_curriculum_strategy():
    """Curriculum 전략 테스트"""
    print("\n=== Curriculum 전략 테스트 ===")
    
    manager = TaskManager(
        domain="dmc_quadruped",
        tasks=["walk", "run", "escape"],
        strategy="curriculum",
        switch_interval=2,
        curriculum_order=["walk", "run", "escape"]
    )
    
    for episode in range(8):
        if manager.should_switch_task():
            manager.switch_task()
        
        print(f"Episode {episode}: {manager.get_current_task_name()}")
        manager.on_episode_end()


def test_config_integration():
    """설정 파일 통합 테스트"""
    print("\n=== 설정 파일 통합 테스트 ===")
    
    # 실제 DreamerV3 설정과 유사한 구조
    class MockConfig:
        def __init__(self):
            self.task_manager = {
                'enabled': True,
                'domain': 'dmc_cartpole',
                'tasks': ['swingup', 'balance'],
                'strategy': 'sequential',
                'switch_interval': 500,
                'random_seed': 42
            }
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = MockConfig()
    
    if config.get('task_manager', {}).get('enabled', False):
        manager = TaskManager(
            domain=config.task_manager['domain'],
            tasks=config.task_manager['tasks'],
            strategy=config.task_manager['strategy'],
            switch_interval=config.task_manager['switch_interval'],
            random_seed=config.task_manager.get('random_seed', 42)
        )
        print(f"Config에서 생성된 TaskManager: {manager.get_stats()}")
    else:
        print("TaskManager가 비활성화됨")


if __name__ == "__main__":
    print("DreamerV3 연속 학습 테스트 시작\n")
    
    test_task_manager()
    test_random_strategy()
    test_curriculum_strategy()
    test_config_integration()
    
    print("\n테스트 완료!")
    
    print("\n=== 사용 방법 ===")
    print("1. 설정 파일에서 task_manager.enabled = True로 설정")
    print("2. domain과 tasks 리스트 지정")
    print("3. strategy 선택 (sequential, random, curriculum)")
    print("4. python -m dreamerv3.main --configs continual_cartpole")