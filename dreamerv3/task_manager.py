import random
import numpy as np
from typing import List, Optional, Dict, Any


class TaskManager:
    """동적 태스크 변경을 위한 매니저 클래스"""
    
    def __init__(
        self,
        domain: str,
        tasks: List[str],
        strategy: str = 'sequential',
        switch_interval: int = 1000,
        **kwargs
    ):
        self.domain = domain
        # Allow CLI to pass tasks as string like "[a,b]" or "a,b",
        # or as a single-element list like ["[a,b]"]
        if isinstance(tasks, str):
            raw = tasks.strip()
            # Remove any surrounding brackets
            if raw.startswith('[') and raw.endswith(']'):
                raw = raw[1:-1]
            # Split on commas and cleanup quotes/brackets
            parts = [p.strip() for p in raw.split(',')] if raw else []
            tasks = [p.strip("'\"").strip('[]') for p in parts if p]
        elif isinstance(tasks, list):
            # Flatten lists that may come from CLI parsing like ["[a", "b", "c]"]
            parsed: list[str] = []
            for item in tasks:
                if not isinstance(item, str):
                    continue
                s = item.strip()
                # Remove stray brackets possibly attached to first/last elements
                s = s.strip('[]')
                # Split further on commas in case a single element encodes multiple tasks
                for piece in s.split(','):
                    piece = piece.strip().strip("'\"").strip('[]')
                    if piece:
                        parsed.append(piece)
            tasks = parsed
        self.tasks = tasks
        self.strategy = strategy
        self.switch_interval = switch_interval
        
        # 상태 관리
        self.current_task_idx = 0
        self.episode_count = 0
        self.step_count = 0
        self.task_history = []
        
        # 전략별 추가 설정
        self.random_seed = kwargs.get('random_seed', 42)
        self.curriculum_order = kwargs.get('curriculum_order', None)
        
        if self.random_seed is not None:
            random.seed(self.random_seed)
            np.random.seed(self.random_seed)
    
    def get_current_task(self) -> str:
        """현재 태스크 반환"""
        if self.strategy == 'sequential':
            return self.tasks[self.current_task_idx % len(self.tasks)]
        elif self.strategy == 'random':
            return random.choice(self.tasks)
        elif self.strategy == 'curriculum':
            return self._get_curriculum_task()
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def get_current_task_name(self) -> str:
        """현재 태스크의 전체 이름 (domain_task) 반환"""
        current_task = self.get_current_task()
        return f"{self.domain}_{current_task}"
    
    def should_switch_task(self) -> bool:
        """태스크 변경 시점인지 확인"""
        return self.episode_count > 0 and self.episode_count % self.switch_interval == 0
    
    def switch_task(self) -> Optional[str]:
        """태스크 변경 실행"""
        old_task = self.get_current_task()
        
        if self.strategy == 'sequential':
            self.current_task_idx = (self.current_task_idx + 1) % len(self.tasks)
        elif self.strategy == 'random':
            # 랜덤은 매번 달라지므로 별도 처리 불필요
            pass
        elif self.strategy == 'curriculum':
            self._advance_curriculum()
        
        new_task = self.get_current_task()
        self.task_history.append({
            'episode': self.episode_count,
            'old_task': old_task,
            'new_task': new_task
        })
        
        print(f"Task switched: {old_task} -> {new_task} (Episode: {self.episode_count})")
        return new_task
    
    def on_episode_start(self):
        """에피소드 시작 시 호출"""
        pass
    
    def on_episode_end(self, episode_reward: float = 0.0, episode_length: int = 0):
        """에피소드 종료 시 호출"""
        self.episode_count += 1
        self.step_count += episode_length
    
    def _get_curriculum_task(self) -> str:
        """커리큘럼 기반 태스크 선택"""
        if self.curriculum_order is None:
            # 기본 커리큘럼: 난이도 순
            return self.tasks[min(self.current_task_idx, len(self.tasks) - 1)]
        else:
            # 사용자 정의 커리큘럼
            idx = min(self.current_task_idx, len(self.curriculum_order) - 1)
            task_name = self.curriculum_order[idx]
            return task_name if task_name in self.tasks else self.tasks[0]
    
    def _advance_curriculum(self):
        """커리큘럼 진행"""
        # 성과 기반으로 다음 태스크로 진행하는 로직
        # 현재는 단순히 순차 진행
        max_idx = len(self.tasks) - 1
        if self.curriculum_order:
            max_idx = len(self.curriculum_order) - 1
        
        self.current_task_idx = min(self.current_task_idx + 1, max_idx)
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 정보 반환"""
        return {
            'current_task': self.get_current_task(),
            'current_task_name': self.get_current_task_name(),
            'episode_count': self.episode_count,
            'step_count': self.step_count,
            'task_history': self.task_history,
            'strategy': self.strategy,
            'switch_interval': self.switch_interval
        }
    
    def reset(self):
        """매니저 상태 초기화"""
        self.current_task_idx = 0
        self.episode_count = 0
        self.step_count = 0
        self.task_history = []


# 사용 예시
if __name__ == "__main__":
    # Sequential 전략
    manager = TaskManager(
        domain="dmc_cartpole",
        tasks=["swingup", "balance", "swingup_sparse"],
        strategy="sequential",
        switch_interval=500
    )
    
    print(f"Current task: {manager.get_current_task_name()}")
    
    # 에피소드 시뮬레이션
    for episode in range(1505):
        manager.on_episode_start()
        
        if manager.should_switch_task():
            manager.switch_task()
        
        manager.on_episode_end(episode_reward=100, episode_length=200)
    
    print(manager.get_stats())