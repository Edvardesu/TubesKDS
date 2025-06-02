import mesa
from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter

from model import SpatialPDModel
from agent import PDAgent, Strategy

def agent_portrayal(agent):
    """
    Fungsi untuk menentukan tampilan agent di visualisasi
    Mengikuti style website dengan warna hijau dan merah
    """
    if agent is None:
        return
    
    portrayal = {
        "Shape": "rect",
        "Filled": "true",
        "w": 1,
        "h": 1,
        "Layer": 0
    }
    
    # Warna berdasarkan strategi (mengikuti website)
    if agent.strategy == Strategy.COOPERATE:
        portrayal["Color"] = "#4CAF50"  # Hijau seperti website
    else:  # DEFECT
        portrayal["Color"] = "#F44336"  # Merah seperti website
        
    return portrayal

class ModelInfoElement(TextElement):
    """
    Element untuk menampilkan informasi model seperti website
    """
    def __init__(self):
        pass
        
    def render(self, model):
        cooperation_rate = model.get_cooperation_rate()
        avg_score = model.get_average_score()
        max_score = model.get_max_score()
        total_agents = model.get_total_agents()
        cooperators = model.get_cooperators_count()
        defectors = model.get_defectors_count()
        
        return f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">📊 Statistics</h3>
            <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                <span><strong>Generation:</strong></span>
                <span>{model.generation}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                <span><strong>Cooperators:</strong></span>
                <span>{cooperators} ({cooperation_rate:.1%})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                <span><strong>Defectors:</strong></span>
                <span>{defectors} ({(1-cooperation_rate):.1%})</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                <span><strong>Avg Score:</strong></span>
                <span>{avg_score:.2f}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 10px 0; padding: 8px; background: #f8f9fa; border-radius: 5px;">
                <span><strong>Max Score:</strong></span>
                <span>{max_score}</span>
            </div>
        </div>
        """

class PayoffInfoElement(TextElement):
    """
    Element untuk menampilkan payoff matrix seperti website
    """
    def __init__(self):
        pass
        
    def render(self, model):
        payoff = model.payoff_matrix
        return f"""
        <div style="background: white; padding: 20px; border-radius: 10px; box-shadow: 0 5px 15px rgba(0,0,0,0.1);">
            <h3 style="margin-top: 0; color: #333; border-bottom: 2px solid #667eea; padding-bottom: 10px;">🎯 Payoff Matrix</h3>
            <table border="1" style="border-collapse: collapse; margin: 10px 0; width: 100%;">
                <tr>
                    <th style="padding: 8px; background: #f8f9fa;"></th>
                    <th style="padding: 8px; background: #f8f9fa;">Cooperate</th>
                    <th style="padding: 8px; background: #f8f9fa;">Defect</th>
                </tr>
                <tr>
                    <th style="padding: 8px; background: #f8f9fa;">Cooperate</th>
                    <td style="padding: 8px; text-align: center;">{payoff[0][0]}</td>
                    <td style="padding: 8px; text-align: center;">{payoff[0][1]}</td>
                </tr>
                <tr>
                    <th style="padding: 8px; background: #f8f9fa;">Defect</th>
                    <td style="padding: 8px; text-align: center;">{payoff[1][0]}</td>
                    <td style="padding: 8px; text-align: center;">{payoff[1][1]}</td>
                </tr>
            </table>
            <p style="font-size: 12px; color: #666;">
                <strong>Update:</strong> {model.update_type.title()} | 
                <strong>Neighborhood:</strong> {model.neighborhood_type.title()}
            </p>
        </div>
        """

class LegendElement(TextElement):
    """
    Element untuk menampilkan legend seperti website
    """
    def __init__(self):
        pass
        
    def render(self, model):
        return """
        <div style="display: flex; justify-content: center; gap: 30px; margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 8px;">
            <div style="display: flex; align-items: center; gap: 10px; font-weight: bold;">
                <div style="width: 20px; height: 20px; background: #4CAF50; border-radius: 4px; border: 1px solid #333;"></div>
                <span>Cooperator</span>
            </div>
            <div style="display: flex; align-items: center; gap: 10px; font-weight: bold;">
                <div style="width: 20px; height: 20px; background: #F44336; border-radius: 4px; border: 1px solid #333;"></div>
                <span>Defector</span>
            </div>
        </div>
        """

# Buat grid visualization dengan ukuran yang dapat disesuaikan
grid = CanvasGrid(agent_portrayal, 50, 50, 500, 500)

# Buat chart untuk cooperation rate
cooperation_chart = ChartModule(
    [{"Label": "Cooperation_Rate", "Color": "#4CAF50"}],
    data_collector_name='datacollector'
)

# Buat chart untuk average score
score_chart = ChartModule(
    [{"Label": "Average_Score", "Color": "#2196F3"}],
    data_collector_name='datacollector'
)

# Buat chart untuk jumlah cooperators vs defectors
population_chart = ChartModule(
    [
        {"Label": "Cooperators", "Color": "#4CAF50"},
        {"Label": "Defectors", "Color": "#F44336"}
    ],
    data_collector_name='datacollector'
)

# Buat chart untuk clustering
clustering_chart = ChartModule(
    [{"Label": "Clustering_Cooperators", "Color": "#9C27B0"}],
    data_collector_name='datacollector'
)

# Elements
model_info = ModelInfoElement()
payoff_info = PayoffInfoElement()
legend_info = LegendElement()

# Parameter yang bisa diubah user (mengikuti website)
model_params = {
    "width": UserSettableParameter(
        "slider",
        "Grid Width",
        50,
        20,
        150,
        5,
        description="Lebar grid"
    ),
    "height": UserSettableParameter(
        "slider", 
        "Grid Height",
        50,
        20,
        150,
        5,
        description="Tinggi grid"
    ),
    "neighborhood_type": UserSettableParameter(
        "choice",
        "Neighborhood Type",
        value="moore",
        choices=["moore", "von_neumann"],
        description="Moore (8 neighbors) atau Von Neumann (4 neighbors)"
    ),
    "update_type": UserSettableParameter(
        "choice",
        "Update Type", 
        value="synchronous",
        choices=["synchronous", "asynchronous"],
        description="Synchronous atau Asynchronous update"
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
    # Payoff Matrix Parameters (mengikuti website)
    "cc_payoff": UserSettableParameter(
        "slider",
        "CC (Both Cooperate)",
        3,
        0,
        10,
        1,
        description="Payoff ketika kedua agent cooperate"
    ),
    "cd_payoff": UserSettableParameter(
        "slider", 
        "CD (I Cooperate, Opponent Defects)",
        0,
        0,
        10,
        1,
        description="Payoff ketika saya cooperate, lawan defect"
    ),
    "dc_payoff": UserSettableParameter(
        "slider",
        "DC (I Defect, Opponent Cooperates)",
        5,
        0,
        10, 
        1,
        description="Payoff ketika saya defect, lawan cooperate"
    ),
    "dd_payoff": UserSettableParameter(
        "slider",
        "DD (Both Defect)",
        1,
        0,
        10,
        1,
        description="Payoff ketika kedua agent defect"
    )
}

def model_creator(**params):
    """
    Fungsi untuk membuat model dengan parameter dari user
    Mengikuti format payoff matrix website
    """
    # Buat payoff matrix dalam format website: [cooperate_row, defect_row]
    payoff_matrix = [
        [params["cc_payoff"], params["cd_payoff"]],  # If I cooperate
        [params["dc_payoff"], params["dd_payoff"]]   # If I defect
    ]
    
    return SpatialPDModel(
        width=params["width"],
        height=params["height"],
        neighborhood_type=params["neighborhood_type"],
        update_type=params["update_type"],
        initial_cooperation_rate=params["initial_cooperation_rate"],
        payoff_matrix=payoff_matrix
    )

# Buat server
server = ModularServer(
    model_creator,
    [
        legend_info,
        grid,
        model_info,
        payoff_info,
        cooperation_chart,
        score_chart,
        population_chart,
        clustering_chart
    ],
    "🎮 Spatial Prisoner's Dilemma",
    model_params
)

# Set port
server.port = 8521

if __name__ == "__main__":
    print("🚀 Starting Spatial Prisoner's Dilemma Server...")
    print("🌐 Open your browser and visit: http://localhost:8521")
    print("⏹️ Press Ctrl+C to stop the server")
    server.launch()