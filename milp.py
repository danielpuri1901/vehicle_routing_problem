import gurobipy as gb
import numpy as np
from _utils import extract_routes


bi = gb.GRB.BINARY
ct = gb.GRB.CONTINUOUS


class ConstructModel(gb.Model):
    def __init__(self, data):
        super().__init__()  # Initialize Gurobi Model
        self.Params.LogFile = "gurobi.log"
        self.Params.Presolve = 2   # Aggressive presolve
        self.Params.Cuts = 2       # Aggressive cut generation
        self._routes = []
        self._dt = data
        self._n = self._dt.args['I'] + 1  # Number of nodes (including depot)
        self._add_vars()
        self._add_constraints()
        self._add_objective()
        model_path = 'output/model_num_vehicle{}_num_customers{}.lp'.format(
            self._dt.args['V'], self._dt.args['I']
        )
        self.write(model_path)
        print(f'Model created at: {model_path}')

    def _add_vars(self):
        self._var = {
            # Binary variables for arcs (i, j) activations
            'x': self.addVars(self._n, self._n, vtype=bi, name='x'),
        }
        # Eliminate self-loops in one batch call instead of n individual sets.
        x = self._var['x']
        self.setAttr("UB", [x[i, i] for i in range(self._n)], 0.0)
        self._preprocess_arcs()
        if self._dt.args['time_windows'] == 1:
            # Auxiliary variables for actual service time (bounded by day schedule)
            self._var['s'] = self.addVars(
                self._n, vtype=ct, name='s',
                lb=self._dt.args['day_start'], ub=self._dt.args['day_end']
            )
        if self._dt.args['capacity'] == 1:
            # Auxiliary variables for cumulative demand (bounded by vehicle capacity)
            self._var['z'] = self.addVars(
                self._n, vtype=ct, name='z',
                lb=0, ub=self._dt.args['v_cap']
            )
        self.update()

    def _preprocess_arcs(self):
        """Set ub=0 on arcs that are provably infeasible given time windows."""
        if self._dt.args['time_windows'] != 1:
            return
        l_t = np.array(self._dt.l_t)      # shape (n-1,)
        e_t = np.array(self._dt.e_t)      # shape (n-1,)
        travel = self._dt.travel_time_matrix
        x = self._var['x']

        # Depot → customer j: unreachable if depot can't arrive before l_j
        depot_arrivals = self._dt.args['day_start'] + travel[0, 1:]
        depot_inf = np.where(depot_arrivals > l_t)[0]
        if len(depot_inf):
            self.setAttr("UB", [x[0, int(j) + 1] for j in depot_inf], 0.0)

        # Customer i → customer j: infeasible if earliest arrival > l_j
        # Use broadcasting: earliest[i, j] = e_t[i] + delta + travel[i+1, j+1]
        earliest = e_t[:, None] + self._dt.args['delta'] + travel[1:, 1:]
        infeasible = earliest > l_t[None, :]
        np.fill_diagonal(infeasible, False)  # i==j handled by self-loop ub
        pairs = np.argwhere(infeasible)
        if len(pairs):
            self.setAttr("UB", [x[int(i) + 1, int(j) + 1] for i, j in pairs], 0.0)

    def _add_constraints(self):
        # Cache frequently-accessed attributes to avoid repeated dict/attr lookups
        # inside O(n^2) generator loops.
        x = self._var['x']
        n = self._n
        dt = self._dt

        # Number of vehicles leaving the depot (at most the entire fleet)
        self.addConstr(
            gb.quicksum(x[0, j] for j in range(1, n)) <= dt.args['V'],
            name='depot_exit'
        )
        # Number of vehicles coming back to the depot (same number that leaves)
        self.addConstr(
            gb.quicksum(x[j, 0] for j in range(1, n)) ==
            gb.quicksum(x[0, j] for j in range(1, n)),
            name='depot_return'
        )
        # Every customer must be satisfied
        self.addConstrs((
            gb.quicksum(x[i, j] for j in range(n) if j != i) == 1
            for i in range(1, n)
        ), name='demand_satisfaction_1')
        self.addConstrs((
            gb.quicksum(x[j, i] for j in range(n) if j != i) == 1
            for i in range(1, n)
        ), name='demand_satisfaction_2')
        if dt.args['capacity'] == 1:
            z = self._var['z']
            demand = dt.demand
            v_cap = dt.args['v_cap']
            # Tight M: max possible load difference between two nodes
            M_cap = v_cap + max(demand)
            # Lower bound: minimum vehicles to cover total demand
            min_vehicles = -(-sum(demand) // v_cap)  # ceil
            self.addConstr(
                gb.quicksum(x[0, j] for j in range(1, n)) >= min_vehicles,
                name='min_fleet_size'
            )
            # Simplified form: z[j] - z[i] - (d + M_cap)*x[i,j] >= -M_cap
            # (algebraically equivalent, one fewer Gurobi term per constraint)
            rhs_cap = -M_cap
            self.addConstrs((
                z[j] - z[i] - (demand[i - 1] + M_cap) * x[i, j] >= rhs_cap
                for i in range(1, n)
                for j in range(1, n)
                if i != j
            ), name='demand_flow_balance')
            # Vehicle capacity constraint (redundant with ub but tightens LP relaxation)
            self.addConstrs((
                z[i] <= v_cap for i in range(n)
            ))
        if dt.args['time_windows'] == 1:
            s = self._var['s']
            travel = dt.travel_time_matrix
            delta = dt.args['delta']
            # Tight M: max time horizon + max customer-to-customer travel + service
            M_time = (
                dt.args['day_end'] - dt.args['day_start']
                + float(np.max(travel[1:, 1:]))
                + delta
            )
            # Simplified form: s[j] - s[i] - (travel[i,j] + M_time)*x[i,j] >= delta - M_time
            rhs_time = delta - M_time
            self.addConstrs((
                s[j] - s[i] - (travel[i, j] + M_time) * x[i, j] >= rhs_time
                for i in range(1, n)
                for j in range(1, n)
                if i != j
            ), name='service_time_balance')
            # Service cannot start before the earliest-start-time
            e_t = dt.e_t
            self.addConstrs((
                s[i] >= e_t[i - 1] for i in range(1, n)
            ))
            # Service cannot start after the latest-start-time
            l_t = dt.l_t
            self.addConstrs((
                s[i] <= l_t[i - 1] for i in range(1, n)
            ))
            self.addConstr(s[0] == dt.args['day_start'], name='day_start_time')
        self.update()

    def _add_objective(self):
        x = self._var['x']
        dist = self._dt.dist_matrix
        n = self._n
        self.setObjective(
            gb.quicksum(
                dist[i, j] * x[i, j]
                for i in range(n)
                for j in range(n)
                if i != j
            ),
            gb.GRB.MINIMIZE
        )
        self.update()

    def _gel_sol(self):
        assert self.status == gb.GRB.OPTIMAL, "Model is not optimal"
        # Fetch all variable values in one Gurobi C-level call
        n = self._n
        x_vals = self.getAttr('X', self._var['x'])
        # Replace n² per-element round() calls with a single vectorized np.rint.
        x_sol = np.rint(
            [x_vals[i, j] for i in range(n) for j in range(n)]
        ).reshape(n, n).astype(int)
        self._sol = {'x': x_sol}

        if self._dt.args['capacity'] == 1:
            z_vals = self.getAttr('X', self._var['z'])
            self._sol['z'] = np.array(
                [round(z_vals[i]) for i in range(self._n)], dtype=int
            )

        if self._dt.args['time_windows'] == 1:
            s_vals = self.getAttr('X', self._var['s'])
            self._sol['s'] = np.array(
                [round(s_vals[i]) for i in range(self._n)], dtype=int
            )

    def solve(self):
        self.optimize()
        self._gel_sol()
        self._routes = extract_routes(self._sol['x'])
        print('\n', '-'*5, 'The Solution', '-'*5, '\n')
        for i, r in enumerate(self._routes):
            print(f'The route vehicle {i} follows is {r}')
        if self._dt.args['time_windows'] == 1:
            print('The service-time solution is:\n', self._sol['s'])
        if self._dt.args['capacity'] == 1:
            print('The vehicle-capacity solution:\n', self._sol['z'])