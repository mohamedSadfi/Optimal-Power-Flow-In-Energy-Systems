
""" Parameters defining the electric system """


params = {
    "Horizon": 1,
    "Unit_Num": 2,
    "Node_Num": 3,
    "Line_Num": 3,
    "SystemLoad_P": 200,
    "SystemLoad_Q": 50,
    "power_factor": 0.95,
    "Gen": [
        {
            "Pmin": 0,
            "Pmax": 150,
            "Qmin": -100,
            "Qmax": 100,
            "Cost_alpha": 0.33,
            "Cost_beta": 30,
            "Cost_gamma": 200,
        },
        {
            "Pmin": 0,
            "Pmax": 50,
            "Qmin": -100,
            "Qmax": 100,
            "Cost_alpha": 0.2,
            "Cost_beta": 20,
            "Cost_gamma": 500,
        },
        {
            "Pmin": 0,
            "Pmax": 50,
            "Qmin": -100,
            "Qmax": 100,
            "Cost_alpha": 0.13,
            "Cost_beta": 40,
            "Cost_gamma": 300,
        },
    ],
    "Wind_P": [50, 60, 70],
    "WP2Node": [1, 0, 0],
    "Network": {
        "Node": [
            {"LoadFactors": 0, "Vmin": 0.95, "Vmax": 1.05},
            {"LoadFactors": 0, "Vmin": 0.95, "Vmax": 1.05},
            {"LoadFactors": 1, "Vmin": 0.95, "Vmax": 1.05},
        ],
        "Line": [
            {
                "Impedance": 0.013 + 1j * 0.13,
                "LineCapacity": 150,
                "StartNode": 1,
                "EndNode": 2,
            },
            {
                "Impedance": 0.013 + 1j * 0.13,
                "LineCapacity": 150,
                "StartNode": 1,
                "EndNode": 3,
            },
            {
                "Impedance": 0.013 + 1j*0.13,
                "LineCapacity": 150,
                "StartNode": 2,
                "EndNode": 3,
            },
        ],
    },
    "refnode": 1,
    "Sbase": 100,
}
