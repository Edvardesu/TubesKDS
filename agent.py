import mesa
import random
from enum import Enum

class Strategy(Enum):
    COOPERATE = 0
    DEFECT = 1

class PDAgent(mesa.Agent):
    """
    Agent untuk Spatial Prisoner's Dilemma yang mengikuti algoritma website
    """
    
    def __init__(self, unique_id, model, x, y, strategy=None):
        super().__init__(unique_id, model)
        
        # Posisi agent di grid
        self.x = x
        self.y = y
        
        # Strategi agent (cooperate atau defect)
        if strategy is None:
            self.strategy = random.choice([Strategy.COOPERATE, Strategy.DEFECT])
        else:
            self.strategy = strategy
            
        # Strategi untuk update berikutnya (untuk synchronous update)
        self.next_strategy = self.strategy
        
        # Skor dari interaksi dalam satu step
        self.score = 0
        
        # Statistik tambahan
        self.total_score = 0
        self.score_history = []
        
    def reset_score(self):
        """Reset skor untuk step baru"""
        self.score = 0
        self.next_strategy = self.strategy
        
    def get_neighbors(self):
        """
        Mendapatkan tetangga berdasarkan tipe neighborhood yang dipilih
        Menggunakan periodic boundary conditions seperti di website
        """
        neighbors = []
        
        if self.model.neighborhood_type == "moore":
            # Moore neighborhood (8 tetangga termasuk diagonal)
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue  # Skip self
                    
                    # Periodic boundary conditions
                    nx = (self.x + dx) % self.model.width
                    ny = (self.y + dy) % self.model.height
                    
                    neighbor = self.model.get_agent_at(nx, ny)
                    if neighbor is not None:
                        neighbors.append(neighbor)
        else:
            # Von Neumann neighborhood (4 tetangga tanpa diagonal)
            directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]
            for dx, dy in directions:
                # Periodic boundary conditions
                nx = (self.x + dx) % self.model.width
                ny = (self.y + dy) % self.model.height
                
                neighbor = self.model.get_agent_at(nx, ny)
                if neighbor is not None:
                    neighbors.append(neighbor)
                    
        return neighbors
        
    def calculate_score(self):
        """
        Menghitung skor berdasarkan interaksi dengan semua tetangga
        Mengikuti algoritma yang sama dengan website
        """
        neighbors = self.get_neighbors()
        self.score = 0
        
        # Interaksi dengan setiap tetangga
        for neighbor in neighbors:
            # Ambil payoff dari model berdasarkan strategi
            my_strategy_idx = self.strategy.value
            neighbor_strategy_idx = neighbor.strategy.value
            
            # Payoff matrix: [my_strategy][neighbor_strategy]
            payoff = self.model.payoff_matrix[my_strategy_idx][neighbor_strategy_idx]
            self.score += payoff
            
    def determine_next_strategy(self):
        """
        Menentukan strategi untuk step berikutnya
        Mengikuti algoritma website: imitasi tetangga dengan skor tertinggi
        """
        neighbors = self.get_neighbors()
        
        # Mulai dengan agent sendiri sebagai yang terbaik
        best_agent = self
        best_score = self.score
        
        # Cari tetangga dengan skor tertinggi
        for neighbor in neighbors:
            if neighbor.score > best_score:
                best_agent = neighbor
                best_score = neighbor.score
                
        # Imitasi strategi terbaik
        self.next_strategy = best_agent.strategy
        
    def update_strategy(self):
        """
        Update strategi untuk synchronous update
        """
        self.strategy = self.next_strategy
        self.total_score += self.score
        self.score_history.append(self.score)
        
    def step(self):
        """
        Step function untuk agent - hanya menghitung skor dan menentukan strategi berikutnya
        Update strategi dilakukan di model level
        """
        self.calculate_score()
        self.determine_next_strategy()
        
    def get_strategy_value(self):
        """
        Mengembalikan nilai strategi sebagai integer (untuk kompatibilitas)
        """
        return self.strategy.value
        
    def set_strategy_from_value(self, value):
        """
        Set strategi dari nilai integer
        """
        if value == 0:
            self.strategy = Strategy.COOPERATE
        else:
            self.strategy = Strategy.DEFECT