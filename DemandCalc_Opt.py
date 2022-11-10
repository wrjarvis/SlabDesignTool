import RebarData
import matplotlib.pyplot as plt
import numpy as np
import math

# Class to define a single slab
class SlabClass:

    # Initial function defines slab geometry
    def __init__(self,
                 primary_dim,
                 secondary_dim,
                 thickness,
                 top_cover=0.06,
                 btm_cover=0.06,
                 side_cover=0.06,
                 mesh=0.2
                 ):

        self.primaryDim = primary_dim
        self.secondaryDim = secondary_dim
        self.meshSize = mesh
        self.thickness = thickness
        self.topCover = top_cover
        self.btmCover = btm_cover
        self.sideCover = side_cover

        self.primaryArray = np.arange(0, self.primaryDim, self.meshSize)
        self.secondaryArray = np.arange(0, self.secondaryDim, self.meshSize)
        self.layerList = ['T1', 'T2', 'B1', 'B2']
        self.rebarDataTable = RebarData.get_dims_table()

        self.requiredDemand = self.calculate_demand()
        self.optimisedDesign = self.optimise_design()
        self.create_rebar_designs()
        self.create_rebar_metrics()
    
    # Function to optimise the design based on a range of factors and requirements
    def optimise_design(self):
        optimisation = []
        check_demands = []
        edge_factors = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5, 6, 7, 8]
        lap_factors = [2, 2.25, 2.5, 2.75, 3, 4, 5, 6, 7, 8]
        extend_factors = [1, 1.25, 1.5, 1.75, 2, 3, 4, 5, 6, 7, 8]
        req_loops = len(edge_factors) * len(lap_factors) * len(extend_factors)
        n = 0
        for edge_factor in edge_factors:
            for lap_factor in lap_factors:
                print(f'Optimisation Progress: {math.floor(n / req_loops * 100)}%')
                for extend_factor in extend_factors:
                    demand = self.optimise_demand(
                        edge_factor=edge_factor,
                        lap_factor=lap_factor,
                        extend_factor=extend_factor
                    )
                    to_add = True
                    for check in check_demands:
                        if np.array_equal(check, demand):
                            to_add = False
                    if to_add:
                        optimisation.append({'Demand': demand})
                        check_demands.append(demand)
                    n += 1
        print('Optimisation Progress: DONE')
        return optimisation

    # Function to find reinforcement metrics for the slab
    def create_rebar_metrics(self):
        for optimised_design in self.optimisedDesign:
            optimised_design['Rebar Metrics'] = {}
            total_weight = 0
            total_n_bars = 0
            total_length = 0
            total_marks = []
            for layer in self.layerList:
                layer_weight = 0
                layer_n_bars = 0
                layer_length = 0
                layer_marks = []
                for rebar in optimised_design['Rebar'][layer]:
                    weight = rebar.weight()
                    length = rebar.length()
                    layer_weight += weight
                    layer_n_bars += 1
                    layer_length += length
                    if length not in layer_marks:
                        layer_marks.append(length)
                    if length not in total_marks:
                        layer_marks.append(length)
                optimised_design['Rebar Metrics'][layer] = {
                    'Total Weight': layer_weight,
                    'Number of Bars': layer_n_bars,
                    'Total Length': layer_length,
                    'Total Bar Marks': len(layer_marks)
                }
                total_weight += layer_weight
                total_n_bars += layer_n_bars
                total_length += layer_length
            optimised_design['Rebar Metrics']['Total'] = {
                    'Total Weight': total_weight,
                    'Number of Bars': total_n_bars,
                    'Total Length': total_length,
                    'Total Bar Marks': len(total_marks)
                }

    # Function to create a set of feasible reinforcement designs
    def create_rebar_designs(self):
        for optimised_design in self.optimisedDesign:
            optimised_design['Rebar'] = {}
            for layer in self.layerList:
                demand = optimised_design['Demand'][layer]
                optimised_design['Rebar'][layer] = self.create_rebar_list(demand, layer)

    #
    def create_rebar_list(self, demand, layer):
        rebar_list = []
        lap_gap = 0.08
        rebar_table = RebarData.get_dims_table()
        if '1' in layer:
            for j, y in enumerate(self.secondaryArray):
                bar_size = None
                start_cords = None
                offset = lap_gap / 2
                for i, x in enumerate(self.primaryArray):
                    if not bar_size:
                        bar_size = demand[i, j]
                        start_cords = [
                            self.primaryArray[i] + self.sideCover,
                            self.secondaryArray[j] + self.meshSize / 2 + offset
                        ]
                    elif demand[i, j] < bar_size:
                        lap_length = rebar_table[demand[i, j]]['lap']
                        end_cords = [
                            self.primaryArray[i] + lap_length,
                            self.secondaryArray[j] + self.meshSize / 2 + offset
                        ]
                        rebar_list.append(RebarClass(start_cords, end_cords, bar_size))
                        offset = -offset
                        bar_size = demand[i, j]
                        start_cords = [
                            self.primaryArray[i],
                            self.secondaryArray[j] + self.meshSize / 2 + offset
                        ]
                    elif bar_size < demand[i, j]:
                        lap_length = rebar_table[bar_size]['lap']
                        end_cords = [
                            self.primaryArray[i],
                            self.secondaryArray[j] + self.meshSize / 2 + offset
                        ]
                        rebar_list.append(RebarClass(start_cords, end_cords, bar_size))
                        offset = -offset
                        bar_size = demand[i, j]
                        start_cords = [
                            self.primaryArray[i] - lap_length,
                            self.secondaryArray[j] + self.meshSize / 2 + offset
                        ]
                end_cords = [
                    max(self.primaryArray) + self.meshSize - self.sideCover,
                    self.secondaryArray[j] + self.meshSize / 2
                ]
                rebar_list.append(RebarClass(start_cords, end_cords, bar_size))
        if '2' in layer:
            for i, y in enumerate(self.primaryArray):
                bar_size = None
                start_cords = None
                offset = lap_gap / 2
                for j, x in enumerate(self.secondaryArray):
                    if not bar_size:
                        bar_size = demand[i, j]
                        start_cords = [
                            self.primaryArray[i] + self.meshSize / 2 + offset,
                            self.secondaryArray[j] + self.sideCover
                        ]
                    elif demand[i, j] < bar_size:
                        lap_length = rebar_table[demand[i, j]]['lap']
                        end_cords = [
                            self.primaryArray[i] + self.meshSize / 2 + offset,
                            self.secondaryArray[j] + lap_length
                        ]
                        rebar_list.append(RebarClass(start_cords, end_cords, bar_size))
                        offset = -offset
                        bar_size = demand[i, j]
                        start_cords = [
                            self.primaryArray[i] + self.meshSize / 2 + offset,
                            self.secondaryArray[j]
                        ]
                    elif bar_size < demand[i, j]:
                        lap_length = rebar_table[bar_size]['lap']
                        end_cords = [
                            self.primaryArray[i] + self.meshSize / 2 + offset,
                            self.secondaryArray[j]
                        ]
                        rebar_list.append(RebarClass(start_cords, end_cords, bar_size))
                        offset = -offset
                        bar_size = demand[i, j]
                        start_cords = [
                            self.primaryArray[i] + self.meshSize / 2 + offset,
                            self.secondaryArray[j] - lap_length
                        ]
                end_cords = [
                    self.primaryArray[i] + self.meshSize / 2,
                    max(self.secondaryArray) + self.meshSize - self.sideCover
                ]
                rebar_list.append(RebarClass(start_cords, end_cords, bar_size))
        return rebar_list

    def optimise_demand(self, edge_factor=1, lap_factor=1, extend_factor=1):
        optimised_demand = self.copy_required_demand()
        optimised_demand = self.update_lap_edges(optimised_demand, edge_factor)
        optimised_demand = self.update_lap_spaces(optimised_demand, lap_factor)
        optimised_demand = self.update_lap_overlap(optimised_demand, extend_factor)
        return optimised_demand

    def update_lap_overlap(self, demand, lap_factor):
        for layer in self.layerList:
            for i, x in enumerate(self.primaryArray):
                for j, y in enumerate(self.secondaryArray):
                    bar_size = demand[layer][i, j]
                    lap_length = self.rebarDataTable[bar_size]['lap']
                    n = math.ceil(lap_length * lap_factor / self.meshSize)
                    if '1' in layer:
                        end_size = None
                        for i_check in range(max(0, i - n), i - 1, 1):
                            if not end_size:
                                end_size = demand[layer][i_check, j]
                            elif demand[layer][i_check, j] != end_size and demand[layer][i_check, j] < bar_size:
                                demand[layer][i_check, j] = bar_size
                        end_size = None
                        for i_check in range(min(len(self.primaryArray) - 1, i + n), i + 1, -1):
                            if not end_size:
                                end_size = demand[layer][i_check, j]
                            elif demand[layer][i_check, j] != end_size and demand[layer][i_check, j] < bar_size:
                                demand[layer][i_check, j] = bar_size
                    elif '2' in layer:
                        end_size = None
                        for j_check in range(max(0, j - n), j - 1, 1):
                            if not end_size:
                                end_size = demand[layer][i, j_check]
                            elif demand[layer][i, j_check] != end_size and demand[layer][i, j_check] < bar_size:
                                demand[layer][i, j_check] = bar_size
                        end_size = None
                        for j_check in range(min(len(self.secondaryArray) - 1, j + n), j + 1, -1):
                            if not end_size:
                                end_size = demand[layer][i, j_check]
                            elif demand[layer][i, j_check] != end_size and demand[layer][i, j_check] < bar_size:
                                demand[layer][i, j_check] = bar_size
        return demand

    def update_lap_spaces(self, demand, lap_factor):
        for layer in self.layerList:
            for i, x in enumerate(self.primaryArray):
                for j, y in enumerate(self.secondaryArray):
                    bar_size = demand[layer][i, j]
                    lap_length = self.rebarDataTable[bar_size]['lap']
                    n = math.ceil(lap_length * lap_factor / self.meshSize)
                    if '1' in layer:
                        update_lap = False
                        for i_check in range(max(0, i - n), i - 1, 1):
                            if demand[layer][i_check, j] == bar_size:
                                update_lap = True
                            elif update_lap and demand[layer][i_check, j] < bar_size:
                                demand[layer][i_check, j] = bar_size
                        update_lap = False
                        for i_check in range(min(len(self.primaryArray) - 1, i + n), i + 1, -1):
                            if demand[layer][i_check, j] == bar_size:
                                update_lap = True
                            elif update_lap and demand[layer][i_check, j] < bar_size:
                                demand[layer][i_check, j] = bar_size
                    elif '2' in layer:
                        update_lap = False
                        for j_check in range(max(0, j - n), j - 1, 1):
                            if demand[layer][i, j_check] == bar_size:
                                update_lap = True
                            elif update_lap and demand[layer][i, j_check] < bar_size:
                                demand[layer][i, j_check] = bar_size
                        update_lap = False
                        for j_check in range(min(len(self.secondaryArray) - 1, j + n), j + 1, -1):
                            if demand[layer][i, j_check] == bar_size:
                                update_lap = True
                            elif update_lap and demand[layer][i, j_check] < bar_size:
                                demand[layer][i, j_check] = bar_size
        return demand

    def update_lap_edges(self, demand, lap_factor):
        for layer in self.layerList:
            for i, x in enumerate(self.primaryArray):
                for j, y in enumerate(self.secondaryArray):
                    bar_size = demand[layer][i, j]
                    lap_length = self.rebarDataTable[bar_size]['lap']
                    n = math.ceil(lap_length * lap_factor / self.meshSize)
                    if i <= n and '1' in layer:
                        for i_check in range(i):
                            if demand[layer][i_check, j] < bar_size:
                                demand[layer][i_check, j] = bar_size
                    elif len(self.primaryArray) - n <= i and '1' in layer:
                        for i_check in range(i, len(self.primaryArray)):
                            if demand[layer][i_check, j] < bar_size:
                                demand[layer][i_check, j] = bar_size
                    if j <= n and '2' in layer:
                        for j_check in range(j):
                            if demand[layer][i, j_check] < bar_size:
                                demand[layer][i, j_check] = bar_size
                    elif len(self.secondaryArray) - n <= j and '2' in layer:
                        for j_check in range(j, len(self.secondaryArray)):
                            if demand[layer][i, j_check] < bar_size:
                                demand[layer][i, j_check] = bar_size
        return demand

    def copy_required_demand(self):
        new_demand = {}
        for layer in self.layerList:
            new_demand[layer] = np.zeros((len(self.primaryArray), len(self.secondaryArray)))
            for i, x in enumerate(self.primaryArray):
                for j, y in enumerate(self.secondaryArray):
                    new_demand[layer][i, j] = self.requiredDemand[layer][i, j]
        return new_demand

    def calculate_demand(self):
        required_demand = {}
        for layer in self.layerList:
            required_demand[layer] = np.zeros((len(self.primaryArray), len(self.secondaryArray)))
        for i, x in enumerate(self.primaryArray):
            for j, y in enumerate(self.secondaryArray):
                x_ratio = x / self.primaryDim
                y_ratio = y / self.secondaryDim
                size = 20
                if 0.3 <= x_ratio <= 0.65 and 0.4 <= y_ratio <= 0.6:
                    size = 32
                elif 0.1 <= x_ratio <= 0.4 and 0.1 <= y_ratio <= 0.2:
                    size = 32
                elif 0.2 <= x_ratio <= 0.7 and 0.3 <= y_ratio <= 0.8:
                    size = 25
                elif 0.1 <= x_ratio <= 0.5 and 0.1 <= y_ratio <= 0.3:
                    size = 25
                elif 0.8 <= x_ratio <= 0.9 and 0.7 <= y_ratio <= 0.9:
                    size = 25
                for layer in self.layerList:
                    required_demand[layer][i, j] = size
        return required_demand

    def plot_slab(self, layer, optimisation=0, plot_rebar=False):
        fig, ax = plt.subplots(1, 2)
        pcm = ax[0].imshow(self.requiredDemand[layer],
                           extent=[0, self.secondaryDim, 0, self.primaryDim],
                           cmap='RdBu_r',
                           aspect='equal',
                           origin='lower')
        ax[0].set_title('Required Demand ' + layer)
        ax[0].set(xlabel='A Dim [m]', ylabel='B Dim [m]')
        fig.colorbar(pcm, ax=ax[0], shrink=0.8)

        pcm = ax[1].imshow(self.optimisedDesign[optimisation]['Demand'][layer],
                           extent=[0, self.secondaryDim, 0, self.primaryDim],
                           cmap='RdBu_r',
                           aspect='equal',
                           origin='lower')
        if plot_rebar:
            for rebar in self.optimisedDesign[optimisation]['Rebar'][layer]:
                primary = [rebar.startCords[0], rebar.endCords[0]]
                secondary = [rebar.startCords[1], rebar.endCords[1]]
                ax[1].plot(secondary, primary, color=rebar.get_colour())
        ax[1].set_title(f'Optimised Demand {layer}\nOptimisation {optimisation}')
        ax[1].set(xlabel='A Dim [m]', ylabel='B Dim [m]')
        fig.colorbar(pcm, ax=ax[1], shrink=0.8)

    def print_optimisations(self, layer=None, print_all=False, print_summary=False, graph=None):
        if layer is None:
            layer = 'Total'
        min_weight = None
        opt_weight = None
        min_bars = None
        opt_bars = None
        min_length = None
        opt_length = None
        min_marks = None
        opt_marks = None
        scatter_x = []
        scatter_y = []
        scatter_list = []
        scatter_annotations = []
        for i, optimised_design in enumerate(self.optimisedDesign):
            if graph is not None:
                x = optimised_design['Rebar Metrics'][layer][graph[0]]
                y = optimised_design['Rebar Metrics'][layer][graph[1]]
                if (x, y) not in scatter_list:
                    scatter_x.append(optimised_design['Rebar Metrics'][layer][graph[0]])
                    scatter_y.append(optimised_design['Rebar Metrics'][layer][graph[1]])
                    scatter_annotations.append(i)
                    scatter_list.append((x, y))
            if print_all:
                print(f'Optimisation: {i}')
                print(optimised_design['Rebar Metrics'][layer])
            if print_summary:
                if i == 0:
                    min_weight = optimised_design['Rebar Metrics'][layer]['Total Weight']
                    opt_weight = i
                    min_bars = optimised_design['Rebar Metrics'][layer]['Number of Bars']
                    opt_bars = i
                    min_length = optimised_design['Rebar Metrics'][layer]['Total Length']
                    opt_length = i
                    min_marks = optimised_design['Rebar Metrics'][layer]['Total Bar Marks']
                    opt_marks = i
                if optimised_design['Rebar Metrics'][layer]['Total Weight'] < min_weight:
                    min_weight = optimised_design['Rebar Metrics'][layer]['Total Weight']
                    opt_weight = i
                if optimised_design['Rebar Metrics'][layer]['Number of Bars'] < min_bars:
                    min_bars = optimised_design['Rebar Metrics'][layer]['Number of Bars']
                    opt_bars = i
                if optimised_design['Rebar Metrics'][layer]['Total Length'] < min_length:
                    min_length = optimised_design['Rebar Metrics'][layer]['Total Length']
                    opt_length = i
                if optimised_design['Rebar Metrics'][layer]['Total Bar Marks'] < min_marks:
                    min_marks = optimised_design['Rebar Metrics'][layer]['Total Bar Marks']
                    opt_marks = i
        if print_summary:
            print(f'Optimised Weight: {opt_weight}, Weight = {min_weight}kg')
            print(f'Optimised Bar Count: {opt_bars}, Bars = {min_bars}')
            print(f'Optimised Length: {opt_length}, Length = {min_length}m')
            print(f'Optimised Bar Mark Count: {opt_marks}, Bar Marks = {min_marks}m')
        if graph is not None:
            plt.scatter(scatter_x, scatter_y)
            for i, txt in enumerate(scatter_annotations):
                plt.annotate(txt, (scatter_x[i], scatter_y[i]))
            plt.title('Optimisation Results')
            plt.xlabel(graph[0])
            plt.ylabel(graph[1])


class RebarClass:
    def __init__(self, start_cords, end_cords, size, shape_code='00'):
        self.startCords = start_cords
        self.endCords = end_cords
        self.barSize = size
        self.shapeCode = shape_code

    def length(self):
        if self.shapeCode == '00':
            dx = self.endCords[0] - self.startCords[0]
            dy = self.endCords[1] - self.startCords[1]
            return (dx ** 2 + dy ** 2) ** 0.5
        return None

    def weight(self):
        mass_per_length = RebarData.get_dims_table()[self.barSize]['weight']
        return mass_per_length * self.length()

    def get_colour(self):
        colours = {
            16: 'green',
            20: 'blue',
            25: 'orange',
            32: 'red',
            40: 'black'
        }
        return colours[self.barSize]


if __name__ == '__main__':

    Slab = SlabClass(15, 12, 0.5)

    Slab.print_optimisations(
        layer='T1',
        print_all=False,
        print_summary=True,
        graph=['Total Weight', 'Number of Bars']
    )
    Slab.plot_slab('T1', plot_rebar=True)
    # Slab.plot_slab(PlotLayer, plot_rebar=True, optimisation=Optimisation)


    '''PlotLayer = 'T2'
    Optimisations = Slab.print_optimisations(layer=PlotLayer, print_all=False, print_summary=True)
    Slab.plot_slab(PlotLayer, plot_rebar=True)
    for Optimisation in Optimisations:
        Slab.plot_slab(PlotLayer, plot_rebar=True, optimisation=Optimisation)'''

