import numpy as np
from scipy.sparse import diags, csc_matrix, eye
from scipy.sparse.linalg import factorized, bicgstab, LinearOperator, spilu


class HeatSolver:
    def __init__(self, physical_constants, simulation_parameters):
        self._is_initialized = False
        self.physical_constants = physical_constants
        self.simulation_parameters = simulation_parameters
        # przypisujemy stałe fizyczne
        self.r = physical_constants["specific_gas_constant_air"]
        self.cp = physical_constants["specific_heat_air_cp"]
        self.p = physical_constants["atmospheric_pressure"]
        self.P = physical_constants["radiator_power"]
        self.alpha_diff = physical_constants["thermal_diffusivity_air"]
        self.alpha_cond = physical_constants["thermal_conductivity_air"]
        self.lambda_concrete = physical_constants["heat_transfer_coefficient_concrete"]
        self.lambda_glass = physical_constants["heat_transfer_coefficient_glass"]
        # przypisujemy parametry symulacji
        self.u_init = simulation_parameters["temp_initial"]
        self.N = simulation_parameters["time_steps"]
        self.h_t = simulation_parameters["time_step"]
        self.h_x = simulation_parameters["space_step"]
        self.L_y = simulation_parameters["room_length"]
        self.L_x = simulation_parameters["room_width"]
        self.u2 = simulation_parameters["temp_external"]
        self.comfort_temp = simulation_parameters["comfort_temp"]
        self.radiator_size_x = simulation_parameters["radiator_size_x"]
        self.radiator_size_y = simulation_parameters["radiator_size_y"]
        self.radiator_pos_x = simulation_parameters["radiator_pos_x"]
        self.radiator_pos_y = simulation_parameters["radiator_pos_y"]
        self.window_pos_x = simulation_parameters["window_pos_x"]
        self.window_width = simulation_parameters["window_width"]
        self.use_iterative_solver = simulation_parameters["use_iterative_solver"]
        self.Nx = int(self.L_x / self.h_x) + 1
        self.Ny = int(self.L_y / self.h_x) + 1
        self.NxNy = self.Nx * self.Ny
        self.solve = None
        self._radiator_coeff = None

        self.iter_tol = 1e-6
        self.iter_maxiter = 500
        self.ilu_drop_tol = 1e-4
        self.ilu_fill_factor = 10
    def run(self):
        """
        tu bedzie petla i bedzie zwracala w sumie najlepiej chyba jakis cache z
        wynikami
        
        :param self: Description
        """
        if not self._is_initialized:
            self.setup_solver()
        # początkowy stan rozwiązania
        u_n = self._u_init_to_vector()
        for t in range(self.N):
            # emituje (zwraca) stan rozwiązania w kroku t bez przerywania pętli
            yield {"step": t, "u_n": u_n.copy()}
            b_n = self._compute_bn_vector(u_n)
            u_next = self._step(b_n)
            u_n = u_next
    
    def _step(self, b_n):
        """
        wykonujemy krok czasowy
        
        :param self: Description
        :param b_n: Description
        """
        if not self._is_initialized:
            raise RuntimeError("Solver not set up. Call setup_solver() first.")
        return self.solve(b_n)

    def setup_solver(self):
        """
        generujemy macierz A i ją faktoryzujemy (stala w czasie)
        """
        self._set_radiator_effect()
        self._precompute_boundary_indices()
        # generujemy macierz drugich pochodnych
        D2 = self._compute_D_2_matrix()
        # tworzymy macierz A
        A = eye(self.NxNy) - self.alpha_diff * self.h_t * D2
        # nakładamy warunki brzegowe
        A = self._apply_boundary_conditions_to_A(A)
        # faktoryzujemy macierz A dla szybkiego rozwiązywania układów równań
        if not self.use_iterative_solver:
            # --- klasyczny, bezpośredni solver (LU) ---
            self.solve = factorized(A.tocsc())
        else:
            self.solve = self._setup_iterative_solver(A)
        # oznaczamy, że inicjalizacja zakończona
        self._is_initialized = True

    def _set_radiator_effect(self):
        j0 = int(self.radiator_pos_x / self.h_x)
        j1 = int((self.radiator_pos_x + self.radiator_size_x) / self.h_x)
        i0 = int(self.radiator_pos_y / self.h_x)
        i1 = int((self.radiator_pos_y + self.radiator_size_y) / self.h_x)

        i = np.arange(i0, i1 + 1)
        j = np.arange(j0, j1 + 1)
        I, J = np.meshgrid(i, j, indexing="ij")
        idx = (I * self.Nx + J).ravel()

        self._radiator_idx = idx.astype(np.int64)

        area_val = self.radiator_size_x * self.radiator_size_y
        coeff = (self.P * self.r) / (self.p * self.cp * area_val)

        # trzymamy sam współczynnik, a nie NxNy-wektor
        self._radiator_coeff = coeff

    def _compute_D_2_matrix(self):
        """
        generuje macierz drugich pochodnych w przestrzeni korzystając z
        metody różnic skończonych (centralnych)
        """
        # liczba punktów siatki w przestrzeni
        inv_h2 = 1.0 / (self.h_x * self.h_x)

        # przekątna główna
        main = (-4.0 * inv_h2) * np.ones(self.NxNy)

        # przekątne ±1 (lewo/prawo) z wycięciem przejść między wierszami
        east = inv_h2 * np.ones(self.NxNy - 1)  # offset +1
        west = inv_h2 * np.ones(self.NxNy - 1)  # offset -1
        # indeksy k będące początkiem wiersza (j=0): k % Nx == 0
        # dla nich połączenie z k-1 byłoby błędne (koniec poprzedniego wiersza)
        cut = (np.arange(1, self.NxNy) % self.Nx == 0)
        west[cut] = 0.0
        east[cut] = 0.0

        # przekątne ±Nx (góra/dół)
        south = inv_h2 * np.ones(self.NxNy - self.Nx)  # offset +Nx
        north = inv_h2 * np.ones(self.NxNy - self.Nx)  # offset -Nx

        D = diags(
            diagonals=[main, east, west, south, north],
            offsets=[0, 1, -1, self.Nx, -self.Nx],
            shape=(self.NxNy, self.NxNy),
            format="csr"
        )
        return D
    
    def _apply_boundary_conditions_to_A(self, A_tilde):
        """
        na już prawie gotową macierz A (A tylda) nakładamy warunki brzegowe
        robina by sie wszystko zgadzalo
        
        :param self: Description
        :param A_tilde: Description
        """
        # trzeba jeszcze wymyslic co zrobic z rogami
        # na razie tylko betonowe ściany zewnętrzne (izolacja)
        c0 = (self.alpha_cond / self.h_x + self.lambda_concrete)
        c1 = -self.alpha_cond / self.h_x
        A_tilde = A_tilde.tolil()  # łatwa podmiana wierszy

        def set_row(k, kin):
            A_tilde.rows[k] = []
            A_tilde.data[k] = []
            A_tilde[k, k] = c0
            A_tilde[k, kin] = c1

        # LEFT edge: j=0, i=1..Ny-2, kin = k+1
        for i in range(0, self.Ny):
            k = i * self.Nx
            set_row(k, k + 1)

        # RIGHT edge: j=Nx-1, i=1..Ny-2, kin = k-1
        for i in range(0, self.Ny):
            k = i * self.Nx + (self.Nx - 1)
            set_row(k, k - 1)

        # TOP edge: i=0, j=1..Nx-2, kin = k+Nx
        for j in range(1, self.Nx - 1):
            k = j
            set_row(k, k + self.Nx)

        # BOTTOM edge: i=Ny-1, j=1..Nx-2, kin = k-Nx
        base = (self.Ny - 1) * self.Nx
        for j in range(1, self.Nx - 1):
            k = base + j
            set_row(k, k - self.Nx)

        return A_tilde.tocsr()
    
    def _apply_boundary_conditions_to_bn(self, b_n):
        """
        nakładamy warunki brzegowe na wektor b_n (prawa strona równania)
        
        :param self: Description
        :param b_n: Description
        """
        # na razie tylko betonowe ściany zewnętrzne (izolacja)
        rhs = self.lambda_concrete * self.u2
        b_n[self._idx_boundary] = rhs
        return b_n
    
    def _compute_bn_vector(self, un):
        b_n = un.copy()

        if self._is_radiator_on(un):
            factor = 1.0 + self.h_t * self._radiator_coeff
            b_n[self._radiator_idx] *= factor

        return self._apply_boundary_conditions_to_bn(b_n)

    def _u_init_to_vector(self):
        """
        zamienia początkowy rozkład temperatury na wektor
        """
        u0 = np.full((self.NxNy,), self.u_init)
        return u0
    
    def _is_radiator_on(self, un):
        """
        sprawdza czy temperatura jest w optymalnym zakresie
        """
        return np.mean(un) <= self.comfort_temp
    
    def _precompute_boundary_indices(self):
        Nx, Ny = self.Nx, self.Ny

        left = np.arange(Ny) * Nx
        right = left + (Nx - 1)
        top = np.arange(Nx)
        bottom = (Ny - 1) * Nx + np.arange(Nx)

        # unikalne, żeby rogi nie dublowały się
        self._idx_boundary = np.unique(np.concatenate([left, right, top, bottom]))

    def _setup_iterative_solver(self, A):
        """
        Przygotowuje solver iteracyjny: BiCGSTAB + (opcjonalnie) ILU jako preconditioner.
        Zwraca funkcję solve(b) -> x.
        """
        A_csc = A.tocsc()

        # Preconditioner ILU (może się wysypać -> fallback bez M)
        try:
            ilu = spilu(
                A_csc,
                drop_tol=self.ilu_drop_tol,
                fill_factor=self.ilu_fill_factor
            )
            M = LinearOperator(A.shape, ilu.solve)
        except Exception:
            M = None

        def solve_iter(b):
            x, info = bicgstab(
                A, b,
                M=M,
                maxiter=self.iter_maxiter
            )

            if info != 0:
                # info > 0: nie osiągnął tolerancji w maxiter
                # info < 0: breakdown numeryczny
                raise RuntimeError(
                    f"bicgstab failed (info={info}). "
                    f"Try: larger ilu_fill_factor, smaller ilu_drop_tol, larger iter_maxiter, "
                    f"or relax iter_tol; also check BC consistency."
                )
            return x

        return solve_iter
