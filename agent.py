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
        
        # Validasi input
        if not isinstance(unique_id, int):
            raise ValueError("unique_id harus berupa integer")
        if model is None:
            raise ValueError("model tidak boleh None")
        
        # Strategi agent (cooperate atau defect)
        if strategy is None:
            self.strategy = random.choice([Strategy.COOPERATE, Strategy.DEFECT])
        elif isinstance(strategy, Strategy):
            self.strategy = strategy
        else:
            # Konversi dari nilai numerik jika diperlukan
            if strategy == 1 or strategy == Strategy.COOPERATE.value:
                self.strategy = Strategy.COOPERATE
            elif strategy == 0 or strategy == Strategy.DEFECT.value:
                self.strategy = Strategy.DEFECT
            else:
                raise ValueError(f"Strategy tidak valid: {strategy}")
            
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
        try:
            if not hasattr(self, 'pos') or self.pos is None:
                return []
                
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
            
            # Filter hanya agent yang valid
            valid_neighbors = [n for n in neighbors if isinstance(n, PDAgent)]
            return valid_neighbors
            
        except Exception as e:
            print(f"Error getting neighbors for agent {self.unique_id}: {e}")
            return []
        
    def play_game(self, other_agent):
        """
        Bermain Prisoner's Dilemma dengan agent lain
        Mengembalikan (skor_self, skor_other)
        """
        try:
            if not isinstance(other_agent, PDAgent):
                raise ValueError("other_agent harus berupa PDAgent")
                
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
            
        except Exception as e:
            print(f"Error in play_game for agent {self.unique_id}: {e}")
            # Return default scores
            return 0, 0
        
    def interact_with_neighbors(self):
        """
        Berinteraksi dengan semua tetangga dan mengumpulkan skor
        """
        try:
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
            
        except Exception as e:
            print(f"Error in interact_with_neighbors for agent {self.unique_id}: {e}")
            self.score = 0
        
    def determine_next_strategy(self):
        """
        Menentukan strategi untuk step berikutnya berdasarkan perbandingan dengan tetangga
        Menggunakan imitasi strategi tetangga terbaik
        """
        try:
            neighbors = self.get_neighbors()
            
            if not neighbors:
                # Jika tidak ada tetangga, tetap dengan strategi saat ini
                self.next_strategy = self.strategy
                return
                
            # Cari tetangga dengan skor tertinggi
            best_score = self.score
            best_strategy = self.strategy
            
            for neighbor in neighbors:
                if isinstance(neighbor, PDAgent) and hasattr(neighbor, 'score'):
                    if neighbor.score > best_score:
                        best_score = neighbor.score
                        best_strategy = neighbor.strategy
                        
            # Imitasi strategi terbaik dengan sedikit noise untuk eksplorasi
            if hasattr(self.model, 'mutation_rate') and self.model.mutation_rate > 0:
                if random.random() < self.model.mutation_rate:
                    # Mutasi: pilih strategi random
                    self.next_strategy = random.choice([Strategy.COOPERATE, Strategy.DEFECT])
                else:
                    self.next_strategy = best_strategy
            else:
                self.next_strategy = best_strategy
                
        except Exception as e:
            print(f"Error in determine_next_strategy for agent {self.unique_id}: {e}")
            # Fallback: tetap dengan strategi saat ini
            self.next_strategy = self.strategy
            
    def update_strategy(self):
        """
        Update strategi untuk synchronous update
        """
        try:
            if hasattr(self, 'next_strategy') and self.next_strategy is not None:
                self.strategy = self.next_strategy
            else:
                print(f"Warning: next_strategy not set for agent {self.unique_id}")
        except Exception as e:
            print(f"Error in update_strategy for agent {self.unique_id}: {e}")
        
    def step(self):
        """
        Step function untuk agent
        """
        try:
            # 1. Berinteraksi dengan tetangga
            self.interact_with_neighbors()
            
            # 2. Tentukan strategi berikutnya
            self.determine_next_strategy()
            
            # 3. Jika asynchronous update, langsung update strategi
            if hasattr(self.model, 'update_type') and self.model.update_type == "asynchronous":
                self.update_strategy()
                
        except Exception as e:
            print(f"Error in step for agent {self.unique_id}: {e}")
            
    def get_cooperation_rate(self):
        """
        Mengembalikan tingkat kooperasi agent ini
        """
        try:
            total_interactions = self.cooperation_count + self.defection_count
            if total_interactions == 0:
                return 0
            return self.cooperation_count / total_interactions
        except Exception as e:
            print(f"Error in get_cooperation_rate for agent {self.unique_id}: {e}")
            return 0
        
    def reset_scores(self):
        """
        Reset skor untuk step baru
        """
        try:
            self.score = 0
        except Exception as e:
            print(f"Error in reset_scores for agent {self.unique_id}: {e}")
    
    def get_strategy_string(self):
        """
        Mengembalikan strategi sebagai string
        """
        return "COOPERATE" if self.strategy == Strategy.COOPERATE else "DEFECT"
    
    def get_neighbor_count(self):
        """
        Mengembalikan jumlah tetangga
        """
        try:
            return len(self.get_neighbors())
        except Exception as e:
            print(f"Error in get_neighbor_count for agent {self.unique_id}: {e}")
            return 0
    
    def get_average_score(self):
        """
        Mengembalikan rata-rata skor dari history
        """
        try:
            if not self.score_history:
                return 0
            return sum(self.score_history) / len(self.score_history)
        except Exception as e:
            print(f"Error in get_average_score for agent {self.unique_id}: {e}")
            return 0
    
    def __str__(self):
        """
        String representation untuk debugging
        """
        try:
            return f"PDAgent({self.unique_id}, {self.get_strategy_string()}, score={self.score})"
        except Exception as e:
            return f"PDAgent({self.unique_id}, ERROR: {e})"
    
    def __repr__(self):
        return self.__str__()