from mesa.model import Model
from mesa.agent import Agent
from mesa.space import MultiGrid
from mesa.time import SimultaneousActivation
from mesa.datacollection import DataCollector

import numpy as np

def euclidean_distance(pos1, pos2):
    return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5

class Celda(Agent):
    def __init__(self, unique_id, model, suciedad: bool = False):
        super().__init__(unique_id, model)
        self.sucia = suciedad


class Mueble(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class EstacionCarga(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)


class RobotLimpieza(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.sig_pos = None
        self.movimientos = 0
        self.carga = 100
        self.pasos_aleatorios_restantes = 0
        self.memoria_celdas_visitadas = set()
        self.total_movimientos = 0
        self.recargas_completas = 0

    umbral_bateria = 33  

    def limpiar_una_celda(self, lista_de_celdas_sucias):
        celda_a_limpiar = self.random.choice(lista_de_celdas_sucias)
        celda_a_limpiar.sucia = False
        self.sig_pos = celda_a_limpiar.pos

    def seleccionar_nueva_pos(self, lista_de_vecinos):
        
        
        celdas_disponibles = [celda for celda in lista_de_vecinos if not isinstance(celda, Mueble) and celda.pos not in self.memoria_celdas_visitadas]

        if not celdas_disponibles:
            
            self.memoria_celdas_visitadas.clear()
            celdas_disponibles = [celda for celda in lista_de_vecinos if not isinstance(celda, Mueble)]

        if self.pasos_aleatorios_restantes > 0:
            self.sig_pos = self.random.choice(celdas_disponibles).pos
            self.pasos_aleatorios_restantes -= 1
            return

        celdas_sucias_cercanas = self.buscar_celdas_sucia(celdas_disponibles)
        if celdas_sucias_cercanas:
            self.sig_pos = self.random.choice(celdas_sucias_cercanas).pos
        elif self.carga <= self.umbral_bateria:
            self.buscar_estacion_carga()
        else:
            self.sig_pos = self.random.choice(celdas_disponibles).pos

        if self.sig_pos == self.pos:
            self.sig_pos = self.random.choice(celdas_disponibles).pos



    def buscar_celdas_sucia(self, lista_de_vecinos):
        celdas_sucias = list()
        for vecino in lista_de_vecinos:
            if isinstance(vecino, Celda) and vecino.sucia:
                celdas_sucias.append(vecino)
        return celdas_sucias

    def buscar_estacion_carga(self):
        estaciones = [agent for agent in self.model.schedule.agents if isinstance(agent, EstacionCarga)]
        
        distancias = [(estacion, euclidean_distance(self.pos, estacion.pos)) for estacion in estaciones]
        
        estacion_cercana = min(distancias, key=lambda x: x[1])[0]
        
        if euclidean_distance(self.pos, estacion_cercana.pos) == 1:
            self.sig_pos = estacion_cercana.pos
            return
        if self.pos[0] < estacion_cercana.pos[0]:
            dx = 1
        elif self.pos[0] > estacion_cercana.pos[0]:
            dx = -1
        else:
            dx = 0

        if self.pos[1] < estacion_cercana.pos[1]:
            dy = 1
        elif self.pos[1] > estacion_cercana.pos[1]:
            dy = -1
        else:
            dy = 0
        siguiente_pos = (self.pos[0] + dx, self.pos[1] + dy)

        contenido_siguiente_pos = self.model.grid.get_cell_list_contents([siguiente_pos])
        if any(isinstance(obj, Mueble) for obj in contenido_siguiente_pos):
            if dx != 0:
                siguiente_pos = (self.pos[0] + dx, self.pos[1])
                if any(isinstance(obj, Mueble) for obj in self.model.grid.get_cell_list_contents([siguiente_pos])):
                    siguiente_pos = self.pos 
            elif dy != 0:
                siguiente_pos = (self.pos[0], self.pos[1] + dy)
                if any(isinstance(obj, Mueble) for obj in self.model.grid.get_cell_list_contents([siguiente_pos])):
                    siguiente_pos = self.pos  
        self.sig_pos = siguiente_pos



    def step(self):
        self.total_movimientos += 1
        if self.carga <= self.umbral_bateria:
            self.buscar_estacion_carga()
            return

        cell_content = self.model.grid.get_cell_list_contents([self.pos])
        if any(isinstance(obj, EstacionCarga) for obj in cell_content) and self.carga < 100:
            return

        vecinos = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)

        vecinos = [vecino for vecino in vecinos if not isinstance(vecino, (Mueble, RobotLimpieza))]

        celdas_sucias = self.buscar_celdas_sucia(vecinos)

        if len(celdas_sucias) == 0:
            self.seleccionar_nueva_pos(vecinos)
        else:
            self.limpiar_una_celda(celdas_sucias)

        if self.sig_pos == self.pos:
            self.pasos_aleatorios_restantes = 2
            self.seleccionar_nueva_pos(vecinos)


    def advance(self):
        if self.carga < 100:
            cell_content = self.model.grid.get_cell_list_contents([self.pos])
            if any(isinstance(obj, EstacionCarga) for obj in cell_content):
   
                self.carga = min(100, self.carga + 25)  
                self.recargas_completas += 1
                return  

        if self.pos == self.sig_pos:
            self.pasos_aleatorios_restantes = 2  # Forzamos 2 pasos aleatorios
            vecinos = self.model.grid.get_neighbors(self.pos, moore=True, include_center=False)
            celdas_disponibles = [vecino for vecino in vecinos if not isinstance(vecino, Mueble)]
            if celdas_disponibles:
                self.sig_pos = self.random.choice(celdas_disponibles).pos

        if self.pos != self.sig_pos:
            self.movimientos += 1

        if self.carga > 0:
            self.carga -= 1
            self.model.grid.move_agent(self, self.sig_pos)
        
            self.memoria_celdas_visitadas.add(self.pos)

        





class Habitacion(Model):
    def __init__(self, M: int, N: int, num_agentes: int = 5, porc_celdas_sucias: float = 0.6, porc_muebles: float = 0.1, modo_pos_inicial: str = 'Fija'):
        self.num_agentes = num_agentes
        self.porc_celdas_sucias = porc_celdas_sucias
        self.porc_muebles = porc_muebles

        self.grid = MultiGrid(M, N, False)
        self.schedule = SimultaneousActivation(self)

        posiciones_disponibles = [pos for _, pos in self.grid.coord_iter()]

        num_muebles = int(M * N * porc_muebles)
        posiciones_muebles = self.random.sample(posiciones_disponibles, k=num_muebles)
        for id, pos in enumerate(posiciones_muebles):
            mueble = Mueble(int(f"{num_agentes}0{id}") + 1, self)
            self.grid.place_agent(mueble, pos)
            posiciones_disponibles.remove(pos)

        self.num_celdas_sucias = int(M * N * porc_celdas_sucias)
        posiciones_celdas_sucias = self.random.sample(posiciones_disponibles, k=self.num_celdas_sucias)
        for id, pos in enumerate(posiciones_disponibles):
            suciedad = pos in posiciones_celdas_sucias
            celda = Celda(int(f"{num_agentes}{id}") + 1, self, suciedad)
            self.grid.place_agent(celda, pos)

        posiciones_estaciones = [
            (M//4, N//4),
            (3*M//4, N//4),
            (M//4, 3*N//4),
            (3*M//4, 3*N//4)
        ]
        for id, pos in enumerate(posiciones_estaciones):
            estacion = EstacionCarga(int(f"{num_agentes}00{id}") + 1, self)
            self.grid.place_agent(estacion, pos)
            self.schedule.add(estacion)

        

        if modo_pos_inicial == 'Aleatoria':
            pos_inicial_robots = self.random.sample(posiciones_disponibles, k=num_agentes)
        else:  # 'Fija'
            pos_inicial_robots = [(1, 1)] * num_agentes

        for id in range(num_agentes):
            robot = RobotLimpieza(id, self)
            self.grid.place_agent(robot, pos_inicial_robots[id])
            self.schedule.add(robot)

        self.datacollector = DataCollector(
            model_reporters={"Grid": get_grid, "Cargas": get_cargas, "CeldasSucias": get_sucias},
        )

    def step(self):
        self.datacollector.collect(self)
        self.schedule.step()
        if self.todoLimpio():
            self.running = False
        

    def todoLimpio(model: Model) -> bool:
        for cell in model.grid.coord_iter():
            cell_content, _ = cell
            for obj in cell_content:
                if isinstance(obj, Celda) and obj.sucia:
                    return False
        return True

def get_grid(model: Model) -> np.ndarray:
    grid = np.zeros((model.grid.width, model.grid.height))
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        x, y = pos
        for obj in cell_content:
            if isinstance(obj, RobotLimpieza):
                grid[x][y] = 2
            elif isinstance(obj, Celda):
                grid[x][y] = int(obj.sucia)
            elif isinstance(obj, EstacionCarga):
                grid[x][y] = 3  
    return grid



def get_cargas(model: Model):
    return [(agent.unique_id, agent.carga) for agent in model.schedule.agents if isinstance(agent, RobotLimpieza)]



def get_sucias(model: Model) -> int:
    sum_sucias = 0
    for cell in model.grid.coord_iter():
        cell_content, pos = cell
        for obj in cell_content:
            if isinstance(obj, Celda) and obj.sucia:
                sum_sucias += 1
    return sum_sucias / model.num_celdas_sucias
