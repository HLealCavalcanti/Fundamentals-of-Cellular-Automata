import numpy as np
import matplotlib.pyplot as plt

# Configurações fixas
GRID_SIZE = 100          # Grade 100x100
BETA = 0.3               # Taxa de transmissão por contato
SIGMA = 0.2              # Probabilidade diária E->I (1/período latente)
GAMMA = 0.1              # Probabilidade diária I->R (1/período infeccioso)
MOVEMENT_PROB = 0.1      # Probabilidade de difusão (movimento para célula vizinha)
STEPS = 100              # Número de dias de simulação
SEED = 42                # Semente aleatória para reprodutibilidade
np.random.seed(SEED)

# ------------------- Funções auxiliares -------------------
# Estados: 0=S, 1=E, 2=I, 3=R

def inicializar_grade(n, n_pacientes_zero=5):
#Cria grade n x n toda suscetível e insere pacientes zero aleatoriamente.
    grade = np.zeros((n, n), dtype=int)
    indices = np.random.choice(n*n, n_pacientes_zero, replace=False)
    grade.flat[indices] = 2
    return grade

def contar_vizinhos_infectados(grade):
# Retorna matriz com número de vizinhos infectados (Moore) para cada célula.
# Usa deslocamento circular para condições periódicas
    infectados = (grade == 2)
    viz = (
        np.roll(infectados, 1, axis=0) +
        np.roll(infectados, -1, axis=0) +
        np.roll(infectados, 1, axis=1) +
        np.roll(infectados, -1, axis=1) +
        np.roll(np.roll(infectados, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(infectados, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(infectados, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(infectados, -1, axis=0), -1, axis=1)
    )
    return viz.astype(int)

def difusao(grade, prob=0.1):
    n = grade.shape[0]
    move_mask = np.random.random((n, n)) < prob
    desloc = np.random.randint(-1, 2, size=(n, n, 2))
    novas_linhas = (np.arange(n)[:, None] + desloc[:, :, 0]) % n
    novas_colunas = (np.arange(n)[None, :] + desloc[:, :, 1]) % n
    # Troca de estados (cuidado com trocas duplas)
    grade_copia = grade.copy()
    ja_trocou = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(n):
            if move_mask[i, j] and not ja_trocou[i, j]:
                ni, nj = novas_linhas[i, j], novas_colunas[i, j]
                if not ja_trocou[ni, nj]:
                    grade[i, j], grade[ni, nj] = grade[ni, nj], grade[i, j]
                    ja_trocou[i, j] = True
                    ja_trocou[ni, nj] = True
    return grade

def passo_seir(grade):
    n_inf = contar_vizinhos_infectados(grade)
    prob_infec = 1 - (1 - BETA) ** n_inf
    sorteio = np.random.random(grade.shape)
    novos_e = (grade == 0) & (sorteio < prob_infec)
    sorteio_ei = np.random.random(grade.shape)
    novos_i = (grade == 1) & (sorteio_ei < SIGMA)
    sorteio_ir = np.random.random(grade.shape)
    novos_r = (grade == 2) & (sorteio_ir < GAMMA)
    grade[novos_e] = 1
    grade[novos_i] = 2
    grade[novos_r] = 3
    return grade

def executar_simulacao(grade_inicial, passos, aplicar_quarentena=False,
                       dia_inicio_quarentena=20, frac_quarentena=0.2):
    grade = grade_inicial.copy()
    historico = {'S': [], 'E': [], 'I': [], 'R': []}
    snapshots = {}
    for dia in range(passos):
        # Aplica quarentena a partir de certo dia: escolhe aleatoriamente células
        if aplicar_quarentena and dia >= dia_inicio_quarentena:
            # A quarentena fixa as células em estado S? Ou as remove? Vamos
            # considerar que a quarentena "bloqueia" células suscetíveis,
            # impedindo-as de transitar. Implementaremos tornando-as imunes
            # temporariamente, mas aqui simplificamos: forçamos algumas células
            # a permanecer em S (não podem mudar).
            mascara_quarentena = np.random.random(grade.shape) < frac_quarentena
            # Salvamos estado original só dos suscetíveis?
            # Para não complicar, faremos: se a célula está em quarentena, ela
            # não se infecta (mas pode estar infectada? Não, quarentena só em
            # suscetíveis no início da medida). No nosso modelo simplificado,
            # células em quarentena são como "bloqueadas" no estado atual.
            # Vamos definir que apenas suscetíveis podem ser colocadas em
            # quarentena (não faz sentido infectado ficar em quarentena e
            # continuar infectando). Para isso, quando a quarentena é
            # decretada, uma fração das células suscetíveis são isoladas e
            # não podem mudar de estado. Isso é feito impedindo a transição.
            pass  # Iremos implementar de forma mais simples: após o passo normal, redefinimos algumas células para S.

        grade = passo_seir(grade)
        grade = difusao(grade, MOVEMENT_PROB)

        if aplicar_quarentena and dia >= dia_inicio_quarentena:
            # Aplicação simples: a cada dia após o início, uma fração das células S
            # volta a ser S (não se infecta). Simula isolamento.
            # Na verdade, quarentena remove suscetíveis do convívio, então eles
            # não participam da dinâmica. Vamos zerar a probabilidade de infecção
            # para essas células. Vamos usar uma máscara fixa a partir do dia 20.
            pass

        # Contagem
        unicos, contagem = np.unique(grade, return_counts=True)
        cont = dict(zip(unicos, contagem))
        for estado, nome in [(0, 'S'), (1, 'E'), (2, 'I'), (3, 'R')]:
            historico[nome].append(cont.get(estado, 0))
        if dia in [30, 60]:
            snapshots[dia] = grade.copy()
    return historico, snapshots

# ------------------- Cenário A: propagação livre -------------------
grade_A = inicializar_grade(GRID_SIZE)
hist_A, snaps_A = executar_simulacao(grade_A, STEPS)

# ------------------- Cenário B: com quarentena -------------------
# Reinicia semente para comparabilidade
np.random.seed(SEED)
grade_B = inicializar_grade(GRID_SIZE)
# Implementação simples de quarentena: a partir do dia 20, 20% das células
# suscetíveis são "protegidas" (forçadas a permanecer S). Vamos fazer com
# que a cada passo, 20% das células S não possam mudar.
def executar_simulacao_quarentena(grade_inicial, passos, dia_inicio=20, frac=0.2):
    grade = grade_inicial.copy()
    hist = {'S': [], 'E': [], 'I': [], 'R': []}
    # Define uma máscara aleatória de quarentena (células que ficarão imunes)
    mascara_quarentena = np.random.random(grade.shape) < frac
    for dia in range(passos):
        # Antes do passo, salvamos os suscetíveis em quarentena
        susc_quarentena = (grade == 0) & mascara_quarentena
        grade = passo_seir(grade)
        grade = difusao(grade, MOVEMENT_PROB)
        # Reverte as células em quarentena para S, impedindo qualquer mudança
        grade[susc_quarentena] = 0
        # Contagem
        unicos, contagem = np.unique(grade, return_counts=True)
        cont = dict(zip(unicos, contagem))
        for estado, nome in [(0, 'S'), (1, 'E'), (2, 'I'), (3, 'R')]:
            hist[nome].append(cont.get(estado, 0))
    return hist

hist_B = executar_simulacao_quarentena(grade_B, STEPS)

# ------------------- Geração das Figuras -------------------
# Figura 1: Cenário A – curva epidêmica + snapshots nos dias 30 e 60
fig1, axes = plt.subplots(1, 3, figsize=(15, 4))
# Curva
dias = range(STEPS)
axes[0].plot(dias, hist_A['S'], label='S')
axes[0].plot(dias, hist_A['E'], label='E')
axes[0].plot(dias, hist_A['I'], label='I')
axes[0].plot(dias, hist_A['R'], label='R')
axes[0].set_xlabel('Dias')
axes[0].set_ylabel('Indivíduos')
axes[0].legend()
axes[0].set_title('Curva epidêmica (cenário livre)')
# Snapshots
im1 = axes[1].imshow(snaps_A[30], cmap='viridis', vmin=0, vmax=3, interpolation='none')
axes[1].set_title('Dia 30')
axes[1].axis('off')
im2 = axes[2].imshow(snaps_A[60], cmap='viridis', vmin=0, vmax=3, interpolation='none')
axes[2].set_title('Dia 60')
axes[2].axis('off')
plt.tight_layout()
plt.savefig('figura1.png', dpi=150)  # Salva no diretório atual

# Figura 2: Comparação entre cenário livre e com quarentena – apenas Infectados
fig2, ax = plt.subplots(figsize=(8, 5))
ax.plot(dias, hist_A['I'], label='Propagação livre')
ax.plot(dias, hist_B['I'], label='Com quarentena (20% isolados a partir do dia 20)')
ax.axvline(x=20, color='gray', linestyle='--', alpha=0.5, label='Início da quarentena')
ax.set_xlabel('Dias')
ax.set_ylabel('Número de Infectados')
ax.legend()
ax.set_title('Efeito da quarentena localizada sobre a epidemia')
plt.tight_layout()
plt.savefig('figura2.png', dpi=150)

plt.show()
