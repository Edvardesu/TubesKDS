import mesa
import numpy as np
import random
from agent import PDAgent, Strategy

class SpatialPDModel(mesa.Model):
    """
    Model untuk Spatial Prisoner's Dilemma yang mengikuti algoritma website
    """
    
    def __init__(self, 
                 width=100, 
                 height=100, 
                 neighborhood_type="moore",  # "moore" atau "von_neumann"
                 update_type="synchronous",  # "synchronous" atau "asynchronous"
                 initial_cooperation_rate=0.5,
                 payoff_matrix=None):
        """
        Inisialisasi model
        
        Args:
            width: Lebar grid
            height: Tinggi grid
            neighborhood_type: Tipe tetangga ("moore" atau "von_neumann")
            update_type: Tipe update ("synchronous" atau "asynchronous")
            initial_cooperation_rate: Tingkat kooperasi awal (0.0-1.0)
            payoff_matrix: Matrix payoff 2D array seperti website
        """
        super().__init__()
        
        # Parameter model
        self.width = width
        self.height = height
        self.neighborhood_type = neighborhood_type
        self.update_type = update_type
        self.initial_cooperation_rate = initial_cooperation_rate
        
        # Payoff matrix dalam format website: [my_strategy][opponent_strategy]
        if payoff_matrix is None:
            # Format: [cooperate_payoffs, defect_payoffs]
            # [CC, CD], [DC, DD]
            self.payoff_matrix = [
                [3, 0],  # If I cooperate: CC=3, CD=0
                [5, 1]   # If I defect: DC=5, DD=1
            ]
        else:
            self.payoff_matrix = payoff_matrix
            
        # Grid 2D untuk menyimpan agents
        self.grid = [[None for _ in range(height)] for _ in range(width)]
        
        # Scheduler - tidak digunakan seperti Mesa tradisional karena kita manage sendiri
        self.schedule = mesa.time.BaseScheduler(self)
        
        # Counter untuk generasi
        self.generation = 0
        
        # List semua agents untuk tracking
        self.agents = []
        
        # Inisialisasi agents
        self._create_agents()
        
        # Data collector untuk mengumpulkan statistik
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Generation": lambda m: m.generation,
                "Cooperation_Rate": self.get_cooperation_rate,
                "Average_Score": self.get_average_score,
                "Max_Score": self.get_max_score,
                "Total_Agents": self.get_total_agents,
                "Cooperators": self.get_cooperators_count,
                "Defectors": self.get_defectors_count,
                "Clustering_Cooperators": self.get_cooperation_clustering
            },
            agent_reporters={
                "Strategy": "get_strategy_value",
                "Score": "score",
                "X": "x",
                "Y": "y"
            }
        )
        
        # Kumpulkan data awal
        self.datacollector.collect(self)
        
        self.running = True
        
    def _create_agents(self):
        """
        Membuat agents dan menempatkannya di grid seperti website
        """
        agent_id = 0
        
        for x in range(self.width):
            for y in range(self.height):
                # Tentukan strategi awal berdasarkan tingkat kooperasi
                if random.random() < self.initial_cooperation_rate:
                    strategy = Strategy.COOPERATE
                else:
                    strategy = Strategy.DEFECT
                    
                # Buat agent
                agent = PDAgent(agent_id, self, x, y, strategy)
                agent_id += 1
                
                # Tempatkan di grid
                self.grid[x][y] = agent
                self.agents.append(agent)
                
    def get_agent_at(self, x, y):
        """
        Mendapatkan agent di posisi (x, y)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return None
        
    def step(self):
        """
        Satu step simulasi mengikuti algoritma website
        """
        # Reset skor semua agents
        for agent in self.agents:
            agent.reset_score()
            
        # Calculate scores untuk semua agents
        for agent in self.agents:
            agent.calculate_score()
            
        # Update strategies berdasarkan tipe update
        if self.update_type == "synchronous":
            self._synchronous_update()
        else:
            self._asynchronous_update()
            
        # Increment generation
        self.generation += 1
        
        # Kumpulkan data
        self.datacollector.collect(self)
        
    def _synchronous_update(self):
        """
        Update synchronous: semua agents menentukan strategi baru, 
        kemudian semua update bersamaan
        """
        # Semua agents menentukan strategi berikutnya
        for agent in self.agents:
            agent.determine_next_strategy()
            
        # Semua agents update strategi bersamaan
        for agent in self.agents:
            agent.update_strategy()
            
    def _asynchronous_update(self):
        """
        Update asynchronous: update subset random agents (10% seperti website)
        """
        # Update 10% dari total agents secara random
        update_count = max(1, int(len(self.agents) * 0.1))
        
        # Pilih agents random untuk update
        agents_to_update = random.sample(self.agents, update_count)
        
        for agent in agents_to_update:
            agent.determine_next_strategy()
            agent.update_strategy()
            
    def get_cooperation_rate(self):
        """
        Mengembalikan tingkat kooperasi global
        """
        if len(self.agents) == 0:
            return 0
            
        cooperators = sum(1 for agent in self.agents 
                         if agent.strategy == Strategy.COOPERATE)
        return cooperators / len(self.agents)
        
    def get_average_score(self):
        """
        Mengembalikan rata-rata skor
        """
        if len(self.agents) == 0:
            return 0
            
        total_score = sum(agent.score for agent in self.agents)
        return total_score / len(self.agents)
        
    def get_max_score(self):
        """
        Mengembalikan skor maksimum
        """
        if len(self.agents) == 0:
            return 0
            
        return max(agent.score for agent in self.agents)
        
    def get_total_agents(self):
        """
        Mengembalikan jumlah total agents
        """
        return len(self.agents)
        
    def get_cooperators_count(self):
        """
        Mengembalikan jumlah cooperators
        """
        return sum(1 for agent in self.agents 
                  if agent.strategy == Strategy.COOPERATE)
                  
    def get_defectors_count(self):
        """
        Mengembalikan jumlah defectors
        """
        return sum(1 for agent in self.agents 
                  if agent.strategy == Strategy.DEFECT)
                  
    def get_cooperation_clustering(self):
        """
        Menghitung clustering coefficient untuk cooperators
        """
        cooperators = [agent for agent in self.agents 
                      if agent.strategy == Strategy.COOPERATE]
        
        if len(cooperators) == 0:
            return 0
            
        total_clustering = 0
        for agent in cooperators:
            neighbors = agent.get_neighbors()
            if len(neighbors) == 0:
                continue
                
            cooperating_neighbors = sum(1 for neighbor in neighbors 
                                      if neighbor.strategy == Strategy.COOPERATE)
            clustering = cooperating_neighbors / len(neighbors)
            total_clustering += clustering
                
        return total_clustering / len(cooperators)
        
    def set_payoff_matrix(self, cc, cd, dc, dd):
        """
        Set payoff matrix dalam format website
        
        Args:
            cc: Payoff untuk mutual cooperation
            cd: Payoff ketika saya cooperate, lawan defect
            dc: Payoff ketika saya defect, lawan cooperate  
            dd: Payoff untuk mutual defection
        """
        self.payoff_matrix = [
            [cc, cd],  # If I cooperate
            [dc, dd]   # If I defect
        ]
        
    def get_spatial_distribution(self):
        """
        Mengembalikan distribusi spasial strategies sebagai numpy array
        """
        grid_data = np.zeros((self.height, self.width))
        
        for x in range(self.width):
            for y in range(self.height):
                agent = self.grid[x][y]
                if agent:
                    grid_data[y][x] = 1 if agent.strategy == Strategy.COOPERATE else 0
                    
        return grid_data
        
    def reset_model(self):
        """
        Reset model ke kondisi awal
        """
        # Clear grid
        self.grid = [[None for _ in range(self.height)] for _ in range(self.width)]
        self.agents = []
        self.generation = 0
        
        # Buat agents baru
        self._create_agents()
        
        # Reset data collector
        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Generation": lambda m: m.generation,
                "Cooperation_Rate": self.get_cooperation_rate,
                "Average_Score": self.get_average_score,
                "Max_Score": self.get_max_score,
                "Total_Agents": self.get_total_agents,
                "Cooperators": self.get_cooperators_count,
                "Defectors": self.get_defectors_count,
                "Clustering_Cooperators": self.get_cooperation_clustering
            },
            agent_reporters={
                "Strategy": "get_strategy_value",
                "Score": "score",
                "X": "x",
                "Y": "y"
            }
        )
        
        # Kumpulkan data awal
        self.datacollector.collect(self)
        
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
            "neighborhood_type": self.neighborhood_type,
            "update_type": self.update_type,
            "initial_cooperation_rate": self.initial_cooperation_rate,
            "payoff_matrix": self.payoff_matrix
        }
        
        import json
        with open(f"{filename}_config.json", "w") as f:
            json.dump(config, f, indent=2, default=str)