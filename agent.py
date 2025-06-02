import mesa
import random
from enum import Enum

class Strategy(Enum):
    COOPERATE = 1
    DEFECT = 0

class PDAgent(mesa.Agent):
    """
    Agent untuk Spatial Prisoner's Dilemma
    """
    
    def __init__(self, unique_id, model, strategy=None):
        super().__init__(unique_id, model)
        
        # Strategi agent (cooperate atau defect)
        if strategy is None:
            self.strategy = random.choice([Strategy.COOPERATE, Strategy.DEFECT])
        else:
            self.strategy = strategy
            
        # Simpan strategi untuk update berikutnya (untuk synchronous update)
        self.next_strategy = self.strategy
        
        # Skor total dari interaksi dalam satu step
        self.score = 0
        self.total_score = 0
        
        # Riwayat skor untuk tracking
        self.score_history = []
        
        # Statistik
        self.cooperation_count = 0
        self.defection_count = 0
        self.interactions_count = 0
        
    def get_neighbors(self):
        """
        Mendapatkan tetangga berdasarkan tipe neighborhood yang dipilih
        """
        if self.model.neighborhood_type == "moore":
            # Moore neighborhood (8 tetangga termasuk diagonal)
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=True, include_center=False
            )
        else:
            # Von Neumann neighborhood (4 tetangga tanpa diagonal)
            neighbors = self.model.grid.get_neighbors(
                self.pos, moore=False, include_center=False
            )
        return neighbors
        
    def play_game(self, other_agent):
        """
        Bermain Prisoner's Dilemma dengan agent lain
        Mengembalikan (skor_self, skor_other)
        """
        # Ambil payoff matrix dari model
        payoff = self.model.payoff_matrix
        
        # Tentukan strategi masing-masing
        my_strategy = self.strategy
        other_strategy = other_agent.strategy
        
        # Hitung skor berdasarkan payoff matrix
        if my_strategy == Strategy.COOPERATE and other_strategy == Strategy.COOPERATE:
            my_score = payoff["CC"][0]  # Reward
            other_score = payoff["CC"][1]
        elif my_strategy == Strategy.COOPERATE and other_strategy == Strategy.DEFECT:
            my_score = payoff["CD"][0]  # Sucker's payoff
            other_score = payoff["CD"][1]  # Temptation
        elif my_strategy == Strategy.DEFECT and other_strategy == Strategy.COOPERATE:
            my_score = payoff["DC"][0]  # Temptation
            other_score = payoff["DC"][1]  # Sucker's payoff
        else:  # Defect vs Defect
            my_score = payoff["DD"][0]  # Punishment
            other_score = payoff["DD"][1]
            
        return my_score, other_score
        
    def interact_with_neighbors(self):
        """
        Berinteraksi dengan semua tetangga dan mengumpulkan skor
        """
        neighbors = self.get_neighbors()
        self.score = 0
        interaction_count = 0
        
        for neighbor in neighbors:
            if isinstance(neighbor, PDAgent):
                my_score, neighbor_score = self.play_game(neighbor)
                self.score += my_score
                
                # Update statistik
                if self.strategy == Strategy.COOPERATE:
                    self.cooperation_count += 1
                else:
                    self.defection_count += 1
                    
                interaction_count += 1
                
        self.interactions_count += interaction_count
        self.total_score += self.score
        self.score_history.append(self.score)
        
    def determine_next_strategy(self):
        """
        Menentukan strategi untuk step berikutnya berdasarkan perbandingan dengan tetangga
        Menggunakan imitasi strategi tetangga terbaik
        """
        neighbors = self.get_neighbors()
        
        if not neighbors:
            # Jika tidak ada tetangga, tetap dengan strategi saat ini
            self.next_strategy = self.strategy
            return
            
        # Cari tetangga dengan skor tertinggi
        best_score = self.score
        best_strategy = self.strategy
        
        for neighbor in neighbors:
            if isinstance(neighbor, PDAgent) and neighbor.score > best_score:
                best_score = neighbor.score
                best_strategy = neighbor.strategy
                
        # Imitasi strategi terbaik dengan sedikit noise untuk eksplorasi
        if self.model.mutation_rate > 0:
            if random.random() < self.model.mutation_rate:
                # Mutasi: pilih strategi random
                self.next_strategy = random.choice([Strategy.COOPERATE, Strategy.DEFECT])
            else:
                self.next_strategy = best_strategy
        else:
            self.next_strategy = best_strategy
            
    def update_strategy(self):
        """
        Update strategi untuk synchronous update
        """
        self.strategy = self.next_strategy
        
    def step(self):
        """
        Step function untuk agent
        """
        # 1. Berinteraksi dengan tetangga
        self.interact_with_neighbors()
        
        # 2. Tentukan strategi berikutnya
        self.determine_next_strategy()
        
        # 3. Jika asynchronous update, langsung update strategi
        if self.model.update_type == "asynchronous":
            self.update_strategy()
            
    def get_cooperation_rate(self):
        """
        Mengembalikan tingkat kooperasi agent ini
        """
        total_interactions = self.cooperation_count + self.defection_count
        if total_interactions == 0:
            return 0
        return self.cooperation_count / total_interactions
        
    def reset_scores(self):
        """
        Reset skor untuk step baru
        """
        self.score = 0