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
        
        # Parameter model
        self.width = width
        self.height = height
        self.density = density
        self.neighborhood_type = neighborhood_type
        self.update_type = update_type
        self.mutation_rate = mutation_rate
        self.initial_cooperation_rate = initial_cooperation_rate
        
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
            
        # Buat grid
        self.grid = mesa.space.MultiGrid(width, height, True)
        
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
        
    def _create_agents(self):
        """
        Membuat agents dan menempatkannya di grid
        """
        total_cells = self.width * self.height
        num_agents = int(total_cells * self.density)
        
        # Buat list semua posisi yang tersedia
        positions = [(x, y) for x in range(self.width) for y in range(self.height)]
        random.shuffle(positions)
        
        # Pilih subset posisi berdasarkan density
        selected_positions = positions[:num_agents]
        
        for pos in selected_positions:
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
            
    def step(self):
        """
        Satu step simulasi
        """
        # Jalankan step untuk semua agents
        self.schedule.step()
        
        # Jika synchronous update, update semua strategi sekaligus
        if self.update_type == "synchronous":
            for agent in self.schedule.agents:
                agent.update_strategy()
                
        # Kumpulkan data
        self.datacollector.collect(self)
        
    def get_cooperation_rate(self):
        """
        Mengembalikan tingkat kooperasi global
        """
        if len(self.schedule.agents) == 0:
            return 0
            
        cooperators = sum(1 for agent in self.schedule.agents 
                         if agent.strategy == Strategy.COOPERATE)
        return cooperators / len(self.schedule.agents)
        
    def get_average_score(self):
        """
        Mengembalikan rata-rata skor
        """
        if len(self.schedule.agents) == 0:
            return 0
            
        total_score = sum(agent.score for agent in self.schedule.agents)
        return total_score / len(self.schedule.agents)
        
    def get_total_agents(self):
        """
        Mengembalikan jumlah total agents
        """
        return len(self.schedule.agents)
        
    def get_cooperators_count(self):
        """
        Mengembalikan jumlah cooperators
        """
        return sum(1 for agent in self.schedule.agents 
                  if agent.strategy == Strategy.COOPERATE)
                  
    def get_defectors_count(self):
        """
        Mengembalikan jumlah defectors
        """
        return sum(1 for agent in self.schedule.agents 
                  if agent.strategy == Strategy.DEFECT)
                  
    def get_average_neighbors(self):
        """
        Mengembalikan rata-rata jumlah tetangga
        """
        if len(self.schedule.agents) == 0:
            return 0
            
        total_neighbors = 0
        for agent in self.schedule.agents:
            neighbors = agent.get_neighbors()
            total_neighbors += len(neighbors)
            
        return total_neighbors / len(self.schedule.agents)
        
    def get_cooperation_clustering(self):
        """
        Menghitung clustering coefficient untuk cooperators
        Mengukur seberapa cenderung cooperators mengelompok
        """
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
        
    def get_score_variance(self):
        """
        Mengembalikan variance dari skor agents
        """
        if len(self.schedule.agents) == 0:
            return 0
            
        scores = [agent.score for agent in self.schedule.agents]
        return np.var(scores)
        
    def get_payoff_summary(self):
        """
        Mengembalikan ringkasan payoff matrix
        """
        return {
            "CC_Reward": self.payoff_matrix["CC"][0],
            "CD_Sucker": self.payoff_matrix["CD"][0],
            "DC_Temptation": self.payoff_matrix["DC"][0],
            "DD_Punishment": self.payoff_matrix["DD"][0]
        }
        
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
        
    def get_spatial_distribution(self):
        """
        Mengembalikan distribusi spasial strategies
        """
        grid_data = np.zeros((self.height, self.width))
        
        for agent in self.schedule.agents:
            if agent.pos:
                x, y = agent.pos
                grid_data[y][x] = 1 if agent.strategy == Strategy.COOPERATE else 0
                
        return grid_data
        
    def save_results(self, filename):
        """
        Simpan hasil simulasi ke file
        """
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