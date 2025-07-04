import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from model import SpatialPDModel
from agent import PDAgent, Strategy

def agent_portrayal(agent):
    """
    Fungsi untuk menentukan tampilan agent di visualisasi
    """
    if agent is None:
        return
    
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.8
    }
    
    # Warna berdasarkan strategi
    if agent.strategy == Strategy.COOPERATE:
        portrayal["Color"] = "blue"
        portrayal["Layer"] = 1
    else:  # DEFECT
        portrayal["Color"] = "red" 
        portrayal["Layer"] = 1
        
    # Ukuran berdasarkan skor (opsional)
    if hasattr(agent, 'score') and agent.score > 0:
        # Normalisasi ukuran berdasarkan skor
        max_possible_score = 40  # Estimasi skor maksimum
        normalized_score = min(agent.score / max_possible_score, 1.0)
        portrayal["r"] = 0.3 + (0.7 * normalized_score)
    
    return portrayal

class ModelInfoElement(TextElement):
    """
    Element untuk menampilkan informasi model
    """
    def __init__(self):
        pass
        
    def render(self, model):
        try:
            cooperation_rate = model.get_cooperation_rate()
            avg_score = model.get_average_score()
            total_agents = model.get_total_agents()
            cooperators = model.get_cooperators_count()
            defectors = model.get_defectors_count()
            
            return f"""
            <h3>Model Information</h3>
            <p><strong>Step:</strong> {model.schedule.steps}</p>
            <p><strong>Total Agents:</strong> {total_agents}</p>
            <p><strong>Cooperators:</strong> {cooperators} ({cooperation_rate:.2%})</p>
            <p><strong>Defectors:</strong> {defectors} ({(1-cooperation_rate):.2%})</p>
            <p><strong>Average Score:</strong> {avg_score:.2f}</p>
            <p><strong>Neighborhood:</strong> {model.neighborhood_type.title()}</p>
            <p><strong>Update Type:</strong> {model.update_type.title()}</p>
            """
        except Exception as e:
            return f"<h3>Model Information</h3><p>Error: {str(e)}</p>"

class PayoffInfoElement(TextElement):
    """
    Element untuk menampilkan payoff matrix
    """
    def __init__(self):
        pass
        
    def render(self, model):
        try:
            payoff = model.payoff_matrix
            return f"""
            <h3>Payoff Matrix</h3>
            <table border="1" style="border-collapse: collapse; margin: 10px 0;">
                <tr>
                    <th></th>
                    <th>Cooperate</th>
                    <th>Defect</th>
                </tr>
                <tr>
                    <th>Cooperate</th>
                    <td>{payoff['CC'][0]}</td>
                    <td>{payoff['CD'][0]}</td>
                </tr>
                <tr>
                    <th>Defect</th>
                    <td>{payoff['DC'][0]}</td>
                    <td>{payoff['DD'][0]}</td>
                </tr>
            </table>
            <p><small>
            CC: Reward | CD: Sucker's Payoff<br>
            DC: Temptation | DD: Punishment
            </small></p>
            """
        except Exception as e:
            return f"<h3>Payoff Matrix</h3><p>Error: {str(e)}</p>"

# Fixed Canvas Grid creation
def create_canvas_grid():
    """
    Membuat CanvasGrid dengan ukuran yang fleksibel
    """
    return CanvasGrid(agent_portrayal, 50, 50, 500, 500)

# Buat chart untuk cooperation rate
cooperation_chart = ChartModule(
    [{"Label": "Cooperation_Rate", "Color": "Blue"}],
    data_collector_name='datacollector'
)

# Buat chart untuk average score
score_chart = ChartModule(
    [{"Label": "Average_Score", "Color": "Green"}],
    data_collector_name='datacollector'
)

# Buat chart untuk jumlah cooperators vs defectors
population_chart = ChartModule(
    [
        {"Label": "Cooperators", "Color": "Blue"},
        {"Label": "Defectors", "Color": "Red"}
    ],
    data_collector_name='datacollector'
)

# Buat chart untuk clustering
clustering_chart = ChartModule(
    [{"Label": "Clustering_Cooperators", "Color": "Purple"}],
    data_collector_name='datacollector'
)

# Model info element
model_info = ModelInfoElement()
payoff_info = PayoffInfoElement()

# Parameter yang bisa diubah user
model_params = {
    "width": UserSettableParameter(
        "slider",
        "Grid Width",
        50,
        10,
        100,
        1,
        description="Lebar grid (10-100)"
    ),
    "height": UserSettableParameter(
        "slider", 
        "Grid Height",
        50,
        10,
        100,
        1,
        description="Tinggi grid (10-100)"
    ),
    "density": UserSettableParameter(
        "slider",
        "Agent Density",
        0.8,
        0.1,
        1.0,
        0.1,
        description="Kepadatan agent (0.1-1.0)"
    ),
    "neighborhood_type": UserSettableParameter(
        "choice",
        "Neighborhood Type",
        value="moore",
        choices=["moore", "von_neumann"],
        description="Tipe tetangga: Moore (8 neighbors) atau Von Neumann (4 neighbors)"
    ),
    "update_type": UserSettableParameter(
        "choice",
        "Update Type", 
        value="synchronous",
        choices=["synchronous", "asynchronous"],
        description="Tipe update: Synchronous (semua bersamaan) atau Asynchronous (random order)"
    ),
    "initial_cooperation_rate": UserSettableParameter(
        "slider",
        "Initial Cooperation Rate",
        0.5,
        0.0,
        1.0,
        0.1,
        description="Tingkat kooperasi awal (0.0-1.0)"
    ),
    "mutation_rate": UserSettableParameter(
        "slider",
        "Mutation Rate",
        0.01,
        0.0,
        0.1,
        0.01,
        description="Tingkat mutasi strategi (0.0-0.1)"
    ),
    # Payoff Matrix Parameters
    "cc_reward": UserSettableParameter(
        "slider",
        "CC Reward (Mutual Cooperation)",
        3,
        0,
        10,
        1,
        description="Payoff untuk mutual cooperation"
    ),
    "cd_sucker": UserSettableParameter(
        "slider", 
        "CD Sucker's Payoff",
        0,
        0,
        10,
        1,
        description="Payoff ketika cooperate vs defect"
    ),
    "dc_temptation": UserSettableParameter(
        "slider",
        "DC Temptation",
        5,
        0,
        10, 
        1,
        description="Payoff ketika defect vs cooperate"
    ),
    "dd_punishment": UserSettableParameter(
        "slider",
        "DD Punishment (Mutual Defection)",
        1,
        0,
        10,
        1,
        description="Payoff untuk mutual defection"
    )
}

# Custom model class yang mengextend SpatialPDModel untuk kompatibilitas
class WebSpatialPDModel(SpatialPDModel):
    """
    Wrapper untuk SpatialPDModel yang kompatibel dengan Mesa web interface
    """
    
    def __init__(self, width, height, density, neighborhood_type, update_type, 
                 initial_cooperation_rate, mutation_rate, cc_reward, cd_sucker, 
                 dc_temptation, dd_punishment):
        
        # Buat payoff matrix dari parameter individual
        payoff_matrix = {
            "CC": [cc_reward, cc_reward],
            "CD": [cd_sucker, dc_temptation],
            "DC": [dc_temptation, cd_sucker],
            "DD": [dd_punishment, dd_punishment]
        }
        
        # Inisialisasi parent class
        super().__init__(
            width=width,
            height=height,
            density=density,
            neighborhood_type=neighborhood_type,
            update_type=update_type,
            initial_cooperation_rate=initial_cooperation_rate,
            mutation_rate=mutation_rate,
            payoff_matrix=payoff_matrix
        )

# Buat grid visualization
grid = create_canvas_grid()

# Buat server dengan model wrapper
server = ModularServer(
    WebSpatialPDModel,
    [
        grid,
        model_info,
        payoff_info,
        cooperation_chart,
        score_chart,
        population_chart,
        clustering_chart
    ],
    "Spatial Prisoner's Dilemma",
    model_params
)

# Set port
server.port = 8521

if __name__ == "__main__":
    server.launch()