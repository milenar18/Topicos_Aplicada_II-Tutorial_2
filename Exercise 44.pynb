import numpy as np              # Biblioteca para operações numéricas e matrizes
import matplotlib.pyplot as plt # Biblioteca para geração de gráficos
import gudhi                   # Biblioteca para topologia algébrica (complexo de Rips, etc.)

# Define um intervalo [0,1] dividido em 200 pontos, usado como escala para complexificação
J = np.linspace(0, 1, 200)

# Gera um conjunto de 200 pontos aleatórios em 4 dimensões (substitua por dados reais se necessário)
data = np.random.rand(200, 4)

# Função que calcula as curvas de Betti a partir de um conjunto de pontos
def GetBettiCurvesFromPointCloud(X, J, dim=2):
    I = 2*J  # Escala aumentada para garantir a cobertura na construção do complexo
    tmax = max(I)  # Tamanho máximo da aresta para a construção do complexo de Rips

    # Cria o complexo de Rips com base nos pontos e no tamanho máximo da aresta
    rips = gudhi.RipsComplex(points=X, max_edge_length=tmax)

    # Constrói a árvore de símplices até a dimensão especificada
    st = rips.create_simplex_tree(max_dimension=dim)

    # Calcula os intervalos de persistência (homologia) usando coeficiente módulo 2
    st.persistence(persistence_dim_max=True, homology_coeff_field=2)

    # Separa os diagramas de persistência por dimensão (0: componentes, 1: ciclos, 2: cavidades)
    Diagrams = [st.persistence_intervals_in_dimension(i) for i in range(dim+1)]

    BettiCurves = []  # Lista para armazenar as curvas de Betti
    step_x = I[1] - I[0]  # Tamanho do passo entre escalas consecutivas

    # Para cada diagrama de persistência (por dimensão)
    for diagram in Diagrams:
        bc = np.zeros(len(I))  # Inicializa vetor de zeros para a curva de Betti
        if diagram.size != 0:  # Verifica se há intervalos de persistência
            # Converte intervalos contínuos para índices discretos no vetor de Betti
            diagram_int = np.clip(np.ceil((diagram[:,:2] - I[0]) / step_x), 0, len(I)).astype(int)
            # Para cada intervalo, incrementa a curva de Betti nas posições correspondentes
            for interval in diagram_int:
                bc[interval[0]:interval[1]] += 1
        # Adiciona a curva de Betti para essa dimensão à lista
        BettiCurves.append(np.reshape(bc, [1, -1]))

    # Retorna as curvas de Betti como uma matriz (dim+1 x len(I))
    return np.reshape(BettiCurves, (dim+1, len(I)))

# Computa as curvas de Betti do conjunto de pontos aleatórios
BettiCurves = GetBettiCurvesFromPointCloud(data, J, dim=2)

# Cria a figura para visualização
plt.figure(figsize=(10,6))

# Plota a curva de Betti 0 (componentes conexas)
plt.plot(J, BettiCurves[0], label=r'$\beta_0$ (componentes conexas)', linewidth=2)

# Plota a curva de Betti 1 (ciclos)
plt.plot(J, BettiCurves[1], label=r'$\beta_1$ (ciclos)', linewidth=2)

# Plota a curva de Betti 2 (cavidades)
plt.plot(J, BettiCurves[2], label=r'$\beta_2$ (cavidades)', linewidth=2)

# Adiciona rótulos e título ao gráfico
plt.xlabel('Escala', fontsize=12)
plt.ylabel('Número de Betti', fontsize=12)
plt.title('Curvas de Betti do complexo de Rips', fontsize=14)

# Adiciona legenda e grade do gráfico
plt.legend()
plt.grid(True)

# Ajusta o layout e exibe o gráfico
plt.tight_layout()
plt.show()
