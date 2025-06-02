import mesa
import numpy as np
import random
from agent import PDAgent, Strategy

class SpatialPDModel(mesa.Model):
    """
    Model untuk Spatial Prisoner's Dilemma
    """
    
    def __init__(self, 
                 width=100, 
                 height=100, 
                 density=1.0,
                 neighborhood_type="moore",  # "moore" atau "von_neumann"
                 update_type="synchronous",  # "synchronous" atau "asynchronous"
                 payoff_matrix=None,
                 mutation_rate=0.01,
                 initial_cooperation_rate=0.5):
        """
        Inisialisasi model
        
        Args:
            width: Lebar grid
            height: Tinggi grid
            density: Kepadatan agent (0.0-1.0)
            neighborhood_type: Tipe tetangga ("moore" atau "von_neumann")
            update_type: Tipe update ("synchronous" atau "asynchronous")
            payoff_matrix: Matrix payoff untuk Prisoner's Dilemma
            mutation_rate: Tingkat mutasi strategi
            initial_cooperation_rate: Tingkat kooperasi awal
        """
        super().__init__()
        
        # Validasi parameter
        if width <= 0 or height <= 0:
            raise ValueError("Width dan height harus positif")
        if not 0 <= density <= 1:
            raise ValueError("Density harus antara 0 dan 1")
        if not 0 <= initial_cooperation_rate <= 1:
            raise ValueError("Initial cooperation rate harus antara 0 dan 1")
        if not 0 <= mutation_rate <= 1:
            raise ValueError("Mutation rate harus antara 0 dan 1")
        
        # Parameter model
        self.width = int(width)
        self.height = int(height)
        self.density = float(density)
        self.neighborhood_type = str(neighborhood_type)
        self.update_type = str(update_type)
        self.mutation_rate = float(mutation_rate)
        self.initial_cooperation_rate = float(initial_cooperation_rate)
        
        # Payoff matrix default untuk Prisoner's Dilemma
        if payoff_matrix is None:
            # Format: {"CC": [reward_player1, reward_player2], ...}
            # CC = Cooperate-Cooperate, CD = Cooperate-Defect, dll
            self.payoff_matrix = {
                "CC": [3, 3],  # Reward (mutual cooperation)
                "CD": [0, 5],  # Sucker's payoff, Temptation
                "DC": [5, 0],  # Temptation, Sucker's payoff
                "DD": [1, 1]   # Punishment (mutual defection)
            }
        else:
            self.payoff_matrix = payoff_matrix
            
        # Validasi payoff matrix
        self._validate_payoff_matrix()
            
        # Buat grid - pastikan ini adalah MultiGrid object
        try:
            self.grid = mesa.space.MultiGrid(self.width, self.height, True)
            # Verifikasi bahwa grid memiliki atribut yang diperlukan
            if not hasattr(self.grid, 'width') or not hasattr(self.grid, 'height'):
                raise RuntimeError("Grid tidak dibuat dengan benar - tidak memiliki atribut width/height")
        except Exception as e:
            print(f"Error creating grid: {e}")
            raise
        
        # Scheduler untuk agent
        if update_type == "synchronous":
            self.schedule = mesa.time.SimultaneousActivation(self)
        else:  # asynchronous
            self.schedule = mesa.time.RandomActivation(self)
            
        # Counter untuk unique ID
        self.agent_id_counter = 0
        
        # Inisialisasi agents
        self._create_agents()
        
        # Data collector untuk mengumpulkan statistik
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Cooperation_Rate": self.get_cooperation_rate,
                "Average_Score": self.get_average_score,
                "Total_Agents": self.get_total_agents,
                "Cooperators": self.get_cooperators_count,
                "Defectors": self.get_defectors_count,
                "Average_Neighbors": self.get_average_neighbors,
                "Clustering_Cooperators": self.get_cooperation_clustering,
                "Score_Variance": self.get_score_variance
            },
            agent_reporters={
                "Strategy": "strategy",
                "Score": "score",
                "Total_Score": "total_score",
                "Cooperation_Rate": "get_cooperation_rate",
                "X": lambda a: a.pos[0] if a.pos else None,
                "Y": lambda a: a.pos[1] if a.pos else None
            }
        )
        
        # Kumpulkan data awal
        self.datacollector.collect(self)
        
        self.running = True
        
    def _validate_payoff_matrix(self):
        """
        Validasi payoff matrix
        """
        required_keys = ["CC", "CD", "DC", "DD"]
        for key in required_keys:
            if key not in self.payoff_matrix:
                raise ValueError(f"Payoff matrix harus memiliki key '{key}'")
            if not isinstance(self.payoff_matrix[key], list) or len(self.payoff_matrix[key]) != 2:
                raise ValueError(f"Payoff matrix['{key}'] harus berupa list dengan 2 elemen")
        
    def _create_agents(self):
        """
        Membuat agents dan menempatkannya di grid
        """
        total_cells = self.width * self.height
        num_agents = int(total_cells * self.density)
        
        if num_agents == 0:
            print("Warning: Tidak ada agents yang dibuat karena density terlalu rendah")
            return
        
        # Buat list semua posisi yang tersedia
        positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(positions)
        
        # Pilih subset posisi berdasarkan density
        selected_positions = positions[:num_agents]
        
        for pos in selected_positions:
            try:
                # Tentukan strategi awal berdasarkan tingkat kooperasi
                if random.random() < self.initial_cooperation_rate:
                    strategy = Strategy.COOPERATE
                else:
                    strategy = Strategy.DEFECT
                    
                # Buat agent
                agent = PDAgent(self.agent_id_counter, self, strategy)
                self.agent_id_counter += 1
                
                # Tempatkan di grid dan scheduler
                self.grid.place_agent(agent, pos)
                self.schedule.add(agent)
            except Exception as e:
                print(f"Error creating agent at position {pos}: {e}")
                continue
                
    def step(self):
        """
        Satu step simulasi
        """
        try:
            # Jalankan step untuk semua agents
            self.schedule.step()
            
            # Jika synchronous update, update semua strategi sekaligus
            if self.update_type == "synchronous":
                for agent in self.schedule.agents:
                    agent.update_strategy()
                    
            # Kumpulkan data
            self.datacollector.collect(self)
        except Exception as e:
            print(f"Error in step: {e}")
            self.running = False
        
    def get_cooperation_rate(self):
        """
        Mengembalikan tingkat kooperasi global
        """
        try:
            if len(self.schedule.agents) == 0:
                return 0
                
            cooperators = sum(1 for agent in self.schedule.agents 
                             if agent.strategy == Strategy.COOPERATE)
            return cooperators / len(self.schedule.agents)
        except Exception as e:
            print(f"Error in get_cooperation_rate: {e}")
            return 0
        
    def get_average_score(self):
        """
        Mengembalikan rata-rata skor
        """
        try:
            if len(self.schedule.agents) == 0:
                return 0
                
            total_score = sum(agent.score for agent in self.schedule.agents)
            return total_score / len(self.schedule.agents)
        except Exception as e:
            print(f"Error in get_average_score: {e}")
            return 0
        
    def get_total_agents(self):
        """
        Mengembalikan jumlah total agents
        """
        try:
            return len(self.schedule.agents)
        except Exception as e:
            print(f"Error in get_total_agents: {e}")
            return 0
        
    def get_cooperators_count(self):
        """
        Mengembalikan jumlah cooperators
        """
        try:
            return sum(1 for agent in self.schedule.agents 
                      if agent.strategy == Strategy.COOPERATE)
        except Exception as e:
            print(f"Error in get_cooperators_count: {e}")
            return 0
                  
    def get_defectors_count(self):
        """
        Mengembalikan jumlah defectors
        """
        try:
            return sum(1 for agent in self.schedule.agents 
                      if agent.strategy == Strategy.DEFECT)
        except Exception as e:
            print(f"Error in get_defectors_count: {e}")
            return 0
                  
    def get_average_neighbors(self):
        """
        Mengembalikan rata-rata jumlah tetangga
        """
        try:
            if len(self.schedule.agents) == 0:
                return 0
                
            total_neighbors = 0
            for agent in self.schedule.agents:
                neighbors = agent.get_neighbors()
                total_neighbors += len(neighbors)
                
            return total_neighbors / len(self.schedule.agents)
        except Exception as e:
            print(f"Error in get_average_neighbors: {e}")
            return 0
        
    def get_cooperation_clustering(self):
        """
        Menghitung clustering coefficient untuk cooperators
        Mengukur seberapa cenderung cooperators mengelompok
        """
        try:
            cooperators = [agent for agent in self.schedule.agents 
                          if agent.strategy == Strategy.COOPERATE]
            
            if len(cooperators) == 0:
                return 0
                
            total_clustering = 0
            for agent in cooperators:
                neighbors = agent.get_neighbors()
                cooperating_neighbors = sum(1 for neighbor in neighbors 
                                          if neighbor.strategy == Strategy.COOPERATE)
                
                if len(neighbors) > 0:
                    clustering = cooperating_neighbors / len(neighbors)
                    total_clustering += clustering
                    
            return total_clustering / len(cooperators)
        except Exception as e:
            print(f"Error in get_cooperation_clustering: {e}")
            return 0
        
    def get_score_variance(self):
        """
        Mengembalikan variance dari skor agents
        """
        try:
            if len(self.schedule.agents) == 0:
                return 0
                
            scores = [agent.score for agent in self.schedule.agents]
            return np.var(scores)
        except Exception as e:
            print(f"Error in get_score_variance: {e}")
            return 0
        
    def get_payoff_summary(self):
        """
        Mengembalikan ringkasan payoff matrix
        """
        try:
            return {
                "CC_Reward": self.payoff_matrix["CC"][0],
                "CD_Sucker": self.payoff_matrix["CD"][0],
                "DC_Temptation": self.payoff_matrix["DC"][0],
                "DD_Punishment": self.payoff_matrix["DD"][0]
            }
        except Exception as e:
            print(f"Error in get_payoff_summary: {e}")
            return {}
        
    def set_payoff_matrix(self, cc, cd, dc, dd):
        """
        Set custom payoff matrix
        
        Args:
            cc: Reward untuk mutual cooperation
            cd: Sucker's payoff (cooperate vs defect)
            dc: Temptation (defect vs cooperate)
            dd: Punishment untuk mutual defection
        """
        self.payoff_matrix = {
            "CC": [cc, cc],
            "CD": [cd, dc],
            "DC": [dc, cd],
            "DD": [dd, dd]
        }
        
    def reset_model(self):
        """
        Reset model ke kondisi awal
        """
        try:
            # Hapus semua agents
            for agent in list(self.schedule.agents):
                self.grid.remove_agent(agent)
                self.schedule.remove(agent)
                
            # Reset counter
            self.agent_id_counter = 0
            
            # Buat agents baru
            self._create_agents()
            
            # Reset data collector
            self.datacollector = mesa.DataCollector(
                model_reporters={
                    "Cooperation_Rate": self.get_cooperation_rate,
                    "Average_Score": self.get_average_score,
                    "Total_Agents": self.get_total_agents,
                    "Cooperators": self.get_cooperators_count,
                    "Defectors": self.get_defectors_count,
                    "Average_Neighbors": self.get_average_neighbors,
                    "Clustering_Cooperators": self.get_cooperation_clustering,
                    "Score_Variance": self.get_score_variance
                },
                agent_reporters={
                    "Strategy": "strategy",
                    "Score": "score",
                    "Total_Score": "total_score",
                    "Cooperation_Rate": "get_cooperation_rate",
                    "X": lambda a: a.pos[0] if a.pos else None,
                    "Y": lambda a: a.pos[1] if a.pos else None
                }
            )
            
            # Kumpulkan data awal
            self.datacollector.collect(self)
        except Exception as e:
            print(f"Error in reset_model: {e}")
        
    def get_spatial_distribution(self):
        """
        Mengembalikan distribusi spasial strategies
        """
        try:
            grid_data = np.zeros((self.height, self.width))
            
            for agent in self.schedule.agents:
                if agent.pos:
                    x, y = agent.pos
                    # Pastikan koordinat dalam batas
                    if 0 <= x < self.width and 0 <= y < self.height:
                        grid_data[y][x] = 1 if agent.strategy == Strategy.COOPERATE else 0
                    
            return grid_data
        except Exception as e:
            print(f"Error in get_spatial_distribution: {e}")
            return np.zeros((self.height, self.width))
        
    def save_results(self, filename):
        """
        Simpan hasil simulasi ke file
        """
        try:
            model_data = self.datacollector.get_model_vars_dataframe()
            agent_data = self.datacollector.get_agent_vars_dataframe()
            
            model_data.to_csv(f"{filename}_model.csv")
            agent_data.to_csv(f"{filename}_agent.csv")
            
            # Simpan juga konfigurasi model
            config = {
                "width": self.width,
                "height": self.height,
                "density": self.density,
                "neighborhood_type": self.neighborhood_type,
                "update_type": self.update_type,
                "mutation_rate": self.mutation_rate,
                "initial_cooperation_rate": self.initial_cooperation_rate,
                "payoff_matrix": self.payoff_matrix
            }
            
            import json
            with open(f"{filename}_config.json", "w") as f:
                json.dump(config, f, indent=2, default=str)
                
            print(f"Results saved to {filename}_model.csv, {filename}_agent.csv, and {filename}_config.json")
        except Exception as e:
            print(f"Error saving results: {e}")

    def __str__(self):
        """
        String representation untuk debugging
        """
        return f"SpatialPDModel({self.width}x{self.height}, {len(self.schedule.agents)} agents, {self.get_cooperation_rate():.2%} cooperation)"