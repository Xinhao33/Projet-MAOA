# compact formulation of the problem in cplex with ring star inequalities
#
from cplex import Cplex, infinity, SparsePair
from cplex.exceptions import CplexError
import math

nb_station = 5

solution_y = {}
solution_x = {}
solution_station = []

def extract_nodes_from_tsp(file_path):
    # Flag to check if we are in the node coordinates section
    in_node_section = False
    nodes = []

    with open(file_path, 'r') as file:
        for line in file:
            # Check if we have reached the node coordinates section
            if 'NODE_COORD_SECTION' in line:
                in_node_section = True
                continue

            # Check for end of file or end of node section
            if 'EOF' in line :
                break

            # Extract nodes if we are in the node coordinates section
            if in_node_section:
                
                parts = line.strip().split()
                if len(parts) == 3:
                    try:
                        # Convert node coordinates to integers
                        node = [int(parts[0]), [int(parts[1]), int(parts[2])]]
                        nodes.append(node)
                    except ValueError:
                        # Handle any lines that don't have proper numeric values
                        print(f"Invalid line in node section: {line.strip()}")

    return nodes

# Function to calculate Euclidean distance between two points
def euclidean_distance(coord1, coord2):
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)

# extract nodes from the tsp file
nodes = extract_nodes_from_tsp('./Instances_TSP/a280.tsp')



def anneau_etoile_optimization(V, E, dij, p):
    global solution_y, solution_x, solution_station

    # Initialize the CPLEX problem
    problem = Cplex()
    problem.objective.set_sense(problem.objective.sense.minimize)

    # Define decision variables
    x_vars = {ij: f'x_{ij[0]}_{ij[1]}' for ij in E}
    y_vars = {(i, j): f'y_{i}_{j}' for i in V for j in V}
    z_vars = {(i, j): f'z_{i}_{j}' for i in V for j in V if i != j}
    
    # Add x and y variables to the problem
    for ij, var_name in x_vars.items():
        problem.variables.add(names=[var_name], lb=[0], ub=[1], types=["B"])
    for ij, var_name in y_vars.items():
        problem.variables.add(names=[var_name], lb=[0], ub=[1], types=["B"])
    for ij, var_name in z_vars.items():
        problem.variables.add(names=[var_name], lb=[0], ub=[p-1], types=["C"])  # Assuming z variables are continuous


    # Objective function
    obj_x = [(x_vars[ij], dij[ij]) for ij in E]  # Tuples of (variable, coefficient) for xij variables
    obj_y = [(y_vars[(i, j)], dij[(i, j)]) for i in V for j in V if (i, j) in dij]  # Tuples for yij variables

    # Combine the components
    obj_combined = obj_x + obj_y

    # Add the objective to the problem
    problem.objective.set_linear(obj_combined)


    # Constraints
    # Placeholder for adding constraints based on the provided equations
    # Constraint to ensure the sum of yii across all vertices equals p
    var_names = [y_vars[(i, i)] for i in V]  # Collecting all yii variables
    coefficients = [1 for _ in V]  # Coefficients for each variable (all 1s)
    problem.linear_constraints.add(
        lin_expr=[SparsePair(ind=var_names, val=coefficients)],
        senses="E",
        rhs=[p]
    )
    
    # Constraint: Sum of yij over j for each i should be 1
    for i in V:
        var_names = [y_vars[(i, j)] for j in V]
        coefficients = [1] * len(V)
        problem.linear_constraints.add(
            lin_expr=[SparsePair(ind=var_names, val=coefficients)],
            senses="E",
            rhs=[1]
        )
    
    # Constraint: yij ≤ yjj for all i, j in V, i != j
    for i in V:
        for j in V:
            if i != j:
                problem.linear_constraints.add(
                    lin_expr=[SparsePair(ind=[y_vars[(i, j)], y_vars[(j, j)]], val=[1, -1])],
                    senses="L",
                    rhs=[0]
                )
    
    # Constraint: Sum of xij over edges incident to i equals 2yii for all i in V
    for i in V:
        # Collect all edges incident to i
        incident_edges = [x_vars[(i, j)] for j in V if (i, j) in E] 
        coefficients = [1] * len(incident_edges)  # Coefficients for xij
        var_names = incident_edges + [y_vars[(i, i)]]
        coefficients += [-2]  # Coefficient for yii
        problem.linear_constraints.add(
            lin_expr=[SparsePair(ind=var_names, val=coefficients)],
            senses="E",
            rhs=[0]
        )

        
    
    # Constraint: Sum of z1j over j in V \ {1} equals p - 1
    var_names = [z_vars[(1, j)] for j in V if j != 1]
    coefficients = [1] * len(var_names)
    problem.linear_constraints.add(
        lin_expr=[SparsePair(ind=var_names, val=coefficients)],
        senses="E",
        rhs=[p - 1]
    )
    
    # Constraint (6): Sum of zji equals sum of zij + yii for all i in V \ {1}
    for i in V:
        if i != 1:
            # Variables for sum of zji
            zji_vars = [z_vars[(j, i)] for j in V if j != i]
            zji_coefficients = [1] * len(zji_vars)

            # Variables for sum of zij
            zij_vars = [z_vars[(i, j)] for j in V if j not in {1, i}]
            zij_coefficients = [1] * len(zij_vars)

            # Combine variables and coefficients for the constraint
            var_names = zji_vars + zij_vars + [y_vars[(i, i)]]
            coefficients = zji_coefficients + [-1] * len(zij_vars) + [-1]

            # Add the constraint
            problem.linear_constraints.add(
                lin_expr=[SparsePair(ind=var_names, val=coefficients)],
                senses="E",
                rhs=[0]
            )
    
    # Constraint (7): zij + zji ≤ (p - 1)xij for all i in V and j in V \ {1, i}
    for i in V:
        for j in V:
            if j != 1 and j != i:
                # Check if the edge (i, j) or (j, i) exists in E
                if (i, j) in E or (j, i) in E:
                    var_names = [z_vars[(i, j)], z_vars[(j, i)], x_vars[(i, j)] if (i, j) in E else x_vars[(j, i)]]
                    coefficients = [1, 1, -(p - 1)]

                    # Add the constraint
                    problem.linear_constraints.add(
                        lin_expr=[SparsePair(ind=var_names, val=coefficients)],
                        senses="L",
                        rhs=[0]
                    )
    
    # Setting y11 = 1
    problem.variables.set_lower_bounds(y_vars[(1, 1)], 1)
    problem.variables.set_upper_bounds(y_vars[(1, 1)], 1)

    # Setting y1j = 0 for all j in V \ {1}
    for j in V:
        if j != 1:
            problem.variables.set_lower_bounds(y_vars[(1, j)], 0)
            problem.variables.set_upper_bounds(y_vars[(1, j)], 0)
    
    # Constraints to ensure xij is symmetric
    for i in V:
        for j in V:
            if i != j and (i, j) in E and (j, i) in E:
                problem.linear_constraints.add(
                    lin_expr=[SparsePair(ind=[x_vars[(i, j)], x_vars[(j, i)]], val=[1, -1])],
                    senses="E",
                    rhs=[0]
                )


    # Solve the problem
    try:
        problem.solve()
    except CplexError as exc:
        print(exc)
        return None

    # Extract the solution
    solution = problem.solution
    if solution.get_status() in [solution.status.MIP_optimal, solution.status.optimal]:
        # Assuming y_values represent the station selection
        y_values = {(i, j): solution.get_values(f'y_{i}_{j}') for i in V for j in V}
        x_values = {(ij): solution.get_values(f'x_{ij[0]}_{ij[1]}') for ij in E}
        z_values = {(i, j): solution.get_values(f'z_{i}_{j}') for i in V for j in V if i != j}
        # Identify and display the stations
        stations = [i for i in V if y_values[(i, i)] == 1]

        # Store the solution
        solution_y = y_values.copy()
        solution_x = x_values.copy()
        solution_station = stations.copy()
        #print("y_values", y_values)
        print("x_values", x_values)
        #print("z_values", z_values)
        print("Stations selected:", stations)
    else:
        print("No optimal solution found.")

       

# Extract Vertices (V)
V = [node[0] for node in nodes]

#print("v", V)

# Create Edges (E) - Assuming a complete graph
E = [(i, j) for i in V for j in V if i != j]

#print("e", E)

# Create a dictionary to map node numbers to their coordinates
node_coordinates = {node[0]: node[1] for node in nodes}

# Calculate Distances (dij)
dij = {}
for edge in E:
    coord1 = node_coordinates[edge[0]]
    coord2 = node_coordinates[edge[1]]
    dij[edge] = euclidean_distance(coord1, coord2)

# Now V, E, and dij are ready to be used in your optimization function

# Call the function with your data
anneau_etoile_optimization(V, E, dij, p=7)

#print("dij", dij)


# plot the solution
import matplotlib.pyplot as plt

def plot_solution(V, E, x_values, y_values,stations_list, node_coordinates):
    print("stations_list", stations_list)
    # Create a plot
    fig, ax = plt.subplots()

    # Plot all nodes
    for node in V:
        x, y = node_coordinates[node]
        if node in stations_list:
            # Highlight station nodes
            print("coucou")
            print("node", node)
            ax.plot(x, y, 'bo', markersize=10, color = "red")  # 'bo' creates a bigger blue dot for stations
        else:
            ax.plot(x, y, 'ko')  # 'ko' creates a black dot for non-station nodes

    # Draw lines for edges with xij = 1
    for (i, j) in E:
        if x_values.get((i, j), 0) == 1.0:
            x1, y1 = node_coordinates[i]
            x2, y2 = node_coordinates[j]
            ax.plot([x1, x2], [y1, y2], 'k-',color="red")  # 'k-' creates a black line for active connections


    # Draw lines for connections with yij = 1 (i != j)
    for (i, j) in E:
        if i != j and y_values.get((i, j), 0) == 1.0:
            x1, y1 = node_coordinates[i]
            x2, y2 = node_coordinates[j]
            ax.plot([x1, x2], [y1, y2], 'k-',color="blue")  # 'k-' creates a black line

    plt.show()

plot_solution(V, E, solution_x,solution_y, solution_station, node_coordinates)

# Call the function with your data
# plot_solution(V, E, x_values, stations_list, node_coordinates)

