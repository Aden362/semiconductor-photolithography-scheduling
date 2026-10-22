# =============================================================================
#  SCHEDULING - PHOTOLITHOGRAPHY WORK AREA (SEMICONDUCTOR MANUFACTURING)
# =============================================================================

import numpy as np 
import pandas as pd
import xlsxwriter
import matplotlib.pyplot as plt 
import io 

# =============================================================================
# SEZIONE 1 ── PARAMETRI E GENERAZIONE DATI DI INPUT
# =============================================================================

np.random.seed(42)

N = 50
M = 4
R = 5
S = 10

rj = np.random.randint(0, 51, size=N)
pj = np.random.randint(5, 21, size=N)
dj = rj + np.random.randint(30, 121, size=N)
wj = np.random.randint(1, 6, size=N) 

"""RETICOLI"""
reticolo_j = np.random.randint(0, R, size=N) 

"""ELIGIBILITÀ MACCHINE"""
eligibility_matrix = np.zeros((N,M), dtype=bool) 

for j in range(N):
    n_elig = np.random.randint(1, M + 1) 
    macchine_elig = np.random.choice(M, n_elig, replace=False)  
    eligibility_matrix[j, macchine_elig] = True 



# =============================================================================
# SEZIONE 2 ── FUNZIONE DI SIMULAZIONE E OBIETTIVO
# =============================================================================

def simulate(seq):
    avm = np.zeros(M, dtype=int) 
    reticolo_su_macchina = np.full(M, -1, dtype=int) 
    reticolo_libero = np.zeros(R, dtype=int) 
    
    STj = np.zeros(N, dtype=int)
    Cj  = np.zeros(N, dtype=int)
    mj  = np.zeros(N, dtype=int)

    setup_ini = np.full(N, -1, dtype=int)
    setup_fin = np.full(N, -1, dtype=int)

    for pos in range(N): 
        job = int(seq[pos])
        ret = int(reticolo_j[job])
        elig = np.where(eligibility_matrix[job])[0]

        """Caso A: stesso reticolo già montato → nessun setup"""
        t_inizio_A = np.maximum(avm[elig], np.maximum(reticolo_libero[ret], rj[job]))

        """Caso B: reticolo diverso → setup anticipatorio (durata S)"""
        t_setup_start = np.maximum(avm[elig], reticolo_libero[ret])
        t_inizio_B    = np.maximum(t_setup_start + S, rj[job])

        stesso_ret = reticolo_su_macchina[elig] == ret
        t_inizio   = np.where(stesso_ret, t_inizio_A, t_inizio_B)

        best_idx  = np.argmin(t_inizio)
        miglior_m = int(elig[best_idx])

        STj[job] = t_inizio[best_idx] #salva nei registri generali il tempo di inizio
        Cj[job]  = STj[job] + pj[job] 
        mj[job]  = miglior_m #salva l'ID della macchina vincitrice
        
        if not stesso_ret[best_idx]:
            s_start = int(max(avm[miglior_m], reticolo_libero[ret]))
            setup_ini[job] = s_start
            setup_fin[job] = s_start + S

        avm[miglior_m]       = Cj[job]
        reticolo_libero[ret] = Cj[job]

        reticolo_su_macchina[reticolo_su_macchina == ret] = -1
        reticolo_su_macchina[miglior_m]                   = ret

    Fj  = Cj - rj
    Tj  = np.maximum(0, Cj - dj)
    Fwj = wj * Fj
    Twj = wj * Tj
    Uj  = (Tj > 0).astype(int)

    return {
        'Cmax': int(np.max(Cj)),
        'F_tot': int(np.sum(Fj)), 'Fw_tot': int(np.sum(Fwj)),
        'T_tot': int(np.sum(Tj)), 'Tw_tot': int(np.sum(Twj)), '#Tardy': int(np.sum(Uj)),
        'STj': STj, 'Cj': Cj, 'Fj': Fj, 'Fwj': Fwj, 
        'Tj': Tj, 'Twj': Twj, 'mj': mj, 'Setup_ini': setup_ini, 'Setup_fin': setup_fin, 'Uj': Uj
    }

def obj(seq): return simulate(seq)['Tw_tot']

# =============================================================================
# SEZIONE 3 ── ALGORITMI
# =============================================================================
def SPT(): return np.argsort(pj)
def LPT(): return np.argsort(-pj)
def EDD(): return np.argsort(dj)
def WSPT(): return np.argsort(pj / wj)
def ERD(): return np.argsort(rj)
def MDD(): return np.argsort(np.maximum(dj, rj + pj))
def WMDD(): return np.argsort(np.maximum(dj, rj + pj) / wj)

def local_search(seq_iniziale, max_iter=400):
    seq_corrente = np.array(seq_iniziale, dtype=int); val_corrente = obj(seq_corrente) 
    iter_stallo = 0
    while iter_stallo < max_iter:
        i, k = np.random.choice(N, 2, replace=False)
        seq_vicina = seq_corrente.copy(); seq_vicina[i], seq_vicina[k] = seq_vicina[k], seq_vicina[i]
        val_vicina = obj(seq_vicina) 
        if val_vicina < val_corrente:
            seq_corrente = seq_vicina; val_corrente = val_vicina; iter_stallo = 0
        else: iter_stallo += 1
    return seq_corrente

def simulated_annealing(seq_iniziale, T_inizio=500.0, alpha=0.99):
    T_fine = 5.0; iter_per_T = 30
    seq_corrente = np.array(seq_iniziale, dtype=int); val_corrente = obj(seq_corrente)
    miglior_seq = seq_corrente.copy(); miglior_val = val_corrente
    T = T_inizio; storia = [miglior_val]
    while T > T_fine:
        for _ in range(iter_per_T):
            i, k = np.random.choice(N, 2, replace=False)
            seq_nuova = seq_corrente.copy(); seq_nuova[i], seq_nuova[k] = seq_nuova[k], seq_nuova[i]
            val_nuova = obj(seq_nuova); delta = val_nuova - val_corrente
            if delta < 0 or np.random.random() < np.exp(-delta / T):
                seq_corrente = seq_nuova; 
                val_corrente = val_nuova
            if val_corrente < miglior_val:
                miglior_val = val_corrente; miglior_seq = seq_corrente.copy()
        storia.append(miglior_val)
        T *= alpha
    return miglior_seq, storia

def crea_individuo():
    ind = np.arange(N);                
    np.random.shuffle(ind);         
    return ind                     

def order_crossover_OX(p1, p2):   

    start = np.random.randint(0, N - 1);        
    end = np.random.randint(start + 1, N)      

    figlio = np.full(N, -1, dtype=int);         
    figlio[start:end + 1] = p1[start:end + 1]   

    figlio[np.where(figlio == -1)[0]] = p2[~np.isin(p2, p1[start:end + 1])]
    return figlio


def mutazione_inversione(individuo, prob=0.15):
    ind = individuo.copy()      

    if np.random.random() < prob:  
        i = np.random.randint(0, N - 1)     
        k = np.random.randint(i + 1, N)  
        ind[i:k + 1] = ind[i:k + 1][::-1]  
    return ind

def selezione_torneo(popolazione, fitness_vals, k=3):
    candidati  = np.random.choice(len(popolazione), k, replace=False)
    vincitore  = candidati[np.argmin(fitness_vals[candidati])]           
    return popolazione[vincitore].copy()

def algoritmo_genetico(dim_pop=60, n_gen=150):

    
    popolazione  = [crea_individuo() for _ in range(dim_pop)]  
    fitness_vals = np.array([obj(ind) for ind in popolazione])  

    # Identifico il miglior individuo iniziale
    miglior_ind = popolazione[np.argmin(fitness_vals)].copy() 
    miglior_val = float(np.min(fitness_vals))                  
    storia_best = [miglior_val]    

    # ── Ciclo evolutivo ───────────────────────────────────────────────────────
    for _ in range(n_gen):

        nuova_pop = [miglior_ind.copy()]   

        while len(nuova_pop) < dim_pop:   

            # 1. SELEZIONE
            g1 = selezione_torneo(popolazione, fitness_vals)
            g2 = selezione_torneo(popolazione, fitness_vals)

            # 2. CROSSOVER
            figlio = order_crossover_OX(g1, g2) if np.random.random() < 0.85 else g1.copy()

            # 3. MUTAZIONE
            nuova_pop.append(mutazione_inversione(figlio))

        # ── Aggiorno popolazione e fitness ───────────────────────────────────
        popolazione  = nuova_pop
        fitness_vals = np.array([obj(ind) for ind in popolazione])

        # ── Aggiorno il best globale se trovo qualcosa di meglio ─────────────
        idx_best = np.argmin(fitness_vals)
        if fitness_vals[idx_best] < miglior_val:
            miglior_val = float(fitness_vals[idx_best])
            miglior_ind = popolazione[idx_best].copy()

        storia_best.append(miglior_val) 

    return miglior_ind, storia_best     

def tabu_search(seq_iniziale, max_iter=300, tabu_tenure=15):
    seq_corrente = np.array(seq_iniziale, dtype=int);
    val_corrente = obj(seq_corrente)

    miglior_seq = seq_corrente.copy(); 
    miglior_val = val_corrente

    tabu_list = {}; 
    storia = [miglior_val]

    for it in range(max_iter):
        miglior_vicino_seq = None
        miglior_vicino_val = np.inf;
        miglior_vicino_mossa = None

        for _ in range(40):
            i, k = sorted(np.random.choice(N, 2, replace=False).tolist())
            mossa = (i, k);
            seq_vicina = seq_corrente.copy();
            seq_vicina[i], seq_vicina[k] = seq_vicina[k], seq_vicina[i]
            val_vicina = obj(seq_vicina)
            
            if (not (mossa in tabu_list and tabu_list[mossa] > it)
                 or val_vicina < miglior_val) and val_vicina < miglior_vicino_val:
                miglior_vicino_val = val_vicina;
                miglior_vicino_seq = seq_vicina;
                miglior_vicino_mossa = mossa

        if miglior_vicino_seq is not None:
            seq_corrente = miglior_vicino_seq;
            val_corrente = miglior_vicino_val;
            tabu_list[miglior_vicino_mossa] = it + tabu_tenure

            if val_corrente < miglior_val:
                miglior_val = val_corrente;
                miglior_seq = seq_corrente.copy()
                
        storia.append(miglior_val)
    return miglior_seq, storia

# =============================================================================
# SEZIONE 4 ── ESECUZIONE ALGORITMI
# =============================================================================
risultati, sequenze = {}, {}
ordine_metodi = ['SPT', 'LPT', 'ERD', 'EDD', 'WSPT', 'MS', 'MDD', 'WMDD', 'TabuSearch', 'SimAnn', 'GeneticAlg']

for nome, seq in {'SPT': SPT(), 'LPT': LPT(), 'ERD': ERD(), 'EDD': EDD(), 'WSPT': WSPT(), 'MDD': MDD(), 'WMDD': WMDD()}.items():
    risultati[nome] = simulate(seq); sequenze[nome] = seq

seq_ls = local_search(sequenze['WSPT']) 
risultati['MS'] = simulate(seq_ls); sequenze['MS'] = seq_ls

seq_ts, storia_ts = tabu_search(sequenze['WSPT'])
risultati['TabuSearch'] = simulate(seq_ts); sequenze['TabuSearch'] = seq_ts

seq_sa, storia_sa = simulated_annealing(sequenze['WSPT'])
risultati['SimAnn'] = simulate(seq_sa); sequenze['SimAnn'] = seq_sa

seq_ga, storia_ga = algoritmo_genetico()
risultati['GeneticAlg'] = simulate(seq_ga); sequenze['GeneticAlg'] = seq_ga

# =============================================================================
# SEZIONE 5 ── CREAZIONE REPORT EXCEL 
# =============================================================================
excel_filename = f"Report_Scheduling_Completo.xlsx"
writer = pd.ExcelWriter(excel_filename, engine='xlsxwriter')
workbook = writer.book
plt.ioff()

# ── DEFINIZIONE DEGLI STILI ──
fmt_title   = workbook.add_format({'bg_color': '#1F497D', 'font_color': 'white', 'bold': True, 'border': 1, 'valign': 'vcenter'})
fmt_header  = workbook.add_format({'bg_color': '#1F497D', 'font_color': 'white', 'bold': True, 'border': 1, 'align': 'center'})
fmt_norm    = workbook.add_format({'border': 1, 'num_format': '#,##0', 'align': 'center'})
fmt_meta    = workbook.add_format({'bg_color': '#E2EFDA', 'border': 1, 'num_format': '#,##0', 'align': 'center'}) 
fmt_best_n  = workbook.add_format({'bg_color': '#A9D08E', 'bold': True, 'border': 1, 'num_format': '#,##0', 'align': 'center'})
fmt_legend  = workbook.add_format({'font_color': '#7F7F7F', 'italic': True, 'valign': 'vcenter'})

# Stili Tabelle Dettaglio (Modificati per il merging)
fmt_tbl_title = workbook.add_format({'bg_color': '#203764', 'font_color': 'white', 'bold': True, 'border': 1, 'align': 'left', 'valign': 'vcenter'})
fmt_tbl_seq   = workbook.add_format({'bg_color': '#4472C4', 'font_color': 'white', 'border': 1, 'align': 'left', 'valign': 'vcenter', 'text_wrap': True})
fmt_tbl_hdr   = workbook.add_format({'bg_color': '#2F5597', 'font_color': 'white', 'bold': True, 'border': 1, 'align': 'center'})
fmt_tbl_norm  = workbook.add_format({'border': 1, 'align': 'center', 'bg_color': '#F2F2F2'})
fmt_tbl_norm_alt = workbook.add_format({'border': 1, 'align': 'center', 'bg_color': '#FFFFFF'})
fmt_tbl_tardy = workbook.add_format({'bg_color': '#FFE699', 'border': 1, 'align': 'center'})

# Stili Gantt
fmt_gantt_hdr   = workbook.add_format({'font_size': 8, 'align': 'center', 'bottom': 1, 'num_format': '0'})
fmt_gantt_mach  = workbook.add_format({'bold': True, 'align': 'right', 'right': 1})
fmt_gantt_setup = workbook.add_format({'bg_color': '#7F7F7F', 'font_color': 'white', 'border': 1, 'align': 'center', 'valign': 'vcenter', 'font_size': 9})
fmt_gantt_grid  = workbook.add_format({'border': 1, 'border_color': '#E0E0E0'})

# Colori predefiniti per i grafici
colori_grafici_hex = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B2', '#937860', '#DA8CA6', '#7F7F7F', '#A6BDDB', '#74C476', '#9E9AC8']

colori_reticoli_hex = ['#4285F4', '#EA4335', '#FBBC04', '#34A853', '#9C27B0']
fmt_gantt_ret = []
for col in colori_reticoli_hex:
    fmt_gantt_ret.append(workbook.add_format({
        'bg_color': col, 'font_color': 'white', 'border': 1, 'border_color': 'black',
        'align': 'center', 'valign': 'vcenter', 'bold': True, 'font_size': 9
    }))

kpi_cols = ['Cmax', 'F_tot', 'Fw_tot', 'T_tot', 'Tw_tot', '#Tardy']
min_vals = {col: min(risultati[alg][col] for alg in ordine_metodi) for col in kpi_cols}

# ── FOGLIO 1: RIEPILOGO KPI ──
ws_riepilogo = workbook.add_worksheet('Riepilogo KPI')
ws_riepilogo.merge_range('A1:G1', f"FINALE - CONFRONTO TUTTI I METODI | N={N} job, M={M} macchine", fmt_title)
ws_riepilogo.write_row('A2', ['Metodo'] + kpi_cols, fmt_header)

row = 2
for alg in ordine_metodi:
    is_meta = alg in ['TabuSearch', 'SimAnn', 'GeneticAlg']
    base_fmt = fmt_meta if is_meta else fmt_norm
    ws_riepilogo.write(row, 0, alg, base_fmt)
    for col_idx, col_name in enumerate(kpi_cols):
        val = risultati[alg][col_name]
        cell_fmt = fmt_best_n if val == min_vals[col_name] else base_fmt
        ws_riepilogo.write_number(row, col_idx + 1, val, cell_fmt)
    row += 1

ws_riepilogo.write('A15', "Legenda: Celle verdi = valore migliore per quel KPI | Righe verde chiaro = metaeuristici", fmt_legend)
ws_riepilogo.set_column('A:A', 14)
ws_riepilogo.set_column('B:G', 11)

# Tre Grafici Nativi Excel (Cmax, Tw_tot, #Tardy) Affiancati
charts_config = [
    (1, 'Cmax (Makespan) per Metodo', 'A17'),
    (5, 'Tw_tot (Weighted Tardiness) per Metodo', 'G17'),
    (6, 'Job in Ritardo (#Tardy) per Metodo', 'O17')
]

for col_pos, col_title, chart_cell in charts_config:
    chart = workbook.add_chart({'type': 'column'})
    chart.add_series({
        'categories': ['Riepilogo KPI', 2, 0, row - 1, 0],
        'values':     ['Riepilogo KPI', 2, col_pos, row - 1, col_pos],
        'gap': 150,
        'data_labels': {'value': True, 'font': {'size': 9}},
        'points': [{'fill': {'color': c}} for c in colori_grafici_hex]
    })
    chart.set_title({'name': col_title, 'name_font': {'size': 10}})
    chart.set_legend({'none': True})
    ws_riepilogo.insert_chart(chart_cell, chart, {'x_scale': 1.1, 'y_scale': 1.1})


# ── FOGLI 2, 3, 4: MIGLIORI, RETICOLI, ELIGIBILITY ──

elig_strings = [", ".join([f"M{m}" for m in np.where(eligibility_matrix[j])[0]]) for j in range(N)]

# 1. Foglio Migliore per KPI
ws_best = workbook.add_worksheet('Migliore per KPI')
ws_best.set_column('A:B', 22)
ws_best.write_row('A1', ['KPI', 'Miglior Algoritmo'], fmt_tbl_hdr)
best_algs = [alg for col, min_v in min_vals.items() for alg in ordine_metodi if risultati[alg][col] == min_v][:len(kpi_cols)]
for idx, (kpi_name, alg_name) in enumerate(zip(kpi_cols, best_algs)):
    fmt_row = fmt_tbl_norm_alt if idx % 2 == 0 else fmt_tbl_norm
    ws_best.write_row(idx + 1, 0, [kpi_name, alg_name], fmt_row)

# 2. Foglio Assegnazione Reticoli
ws_ret = workbook.add_worksheet('Assegnazione Reticoli')
ws_ret.set_column('A:B', 18)
ws_ret.write_row('A1', ['Job', 'Reticolo Assegnato'], fmt_tbl_hdr)
for idx in range(N):
    fmt_row = fmt_tbl_norm_alt if idx % 2 == 0 else fmt_tbl_norm
    ws_ret.write_row(idx + 1, 0, [idx, reticolo_j[idx]], fmt_row)

# 3. Foglio Eligibility Macchine
ws_elig = workbook.add_worksheet('Eligibility Macchine')
ws_elig.set_column('A:B', 22)
ws_elig.write_row('A1', ['Job', 'Macchine Eleggibili'], fmt_tbl_hdr)
for idx in range(N):
    fmt_row = fmt_tbl_norm_alt if idx % 2 == 0 else fmt_tbl_norm
    ws_elig.write_row(idx + 1, 0, [idx, elig_strings[idx]], fmt_row)

# ── FOGLIO 5: CONVERGENZA META ──
ws_conv = workbook.add_worksheet('Convergenza')
ws_conv.write('A1', "Andamento Funzione Obiettivo", workbook.add_format({'bold': True, 'size': 14}))

for idx, (nome, storia, color) in enumerate([('SimAnn', storia_sa, '#4C9BE8'), ('TabuSearch', storia_ts, '#26A69A'), ('GeneticAlg', storia_ga, '#E87B3E')]):
    fig, ax = plt.subplots(figsize=(8, 4))
    
    ax.plot(storia, color=color, linewidth=2, marker='o', markersize=5, markerfacecolor='white', markeredgewidth=1.5)
    ax.set_title(f'Convergenza – {nome}', fontsize=12, fontweight='bold', color='#333333')

    # --- MODIFICA QUI: Condizione per l'asse X ---
    if nome == 'SimAnn':
        ax.set_xlabel('Livello di temperatura', fontsize=10)
    elif nome == 'GeneticAlg':
        ax.set_xlabel('Generazioni', fontsize=10)
    elif nome == 'TabuSearch':
        ax.set_xlabel('Iterazioni', fontsize=10)
    # ---------------------------------------------
    ax.set_ylabel('Tw_tot (Tardiness Pesata)', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    img_buf = io.BytesIO()
    plt.savefig(img_buf, format='png', dpi=100)
    img_buf.seek(0)
    ws_conv.insert_image(idx * 22 + 2, 1, f'conv_{nome}.png', {'image_data': img_buf})
    plt.close()

# ── FOGLI 6+: DETTAGLIO ALGORITMI (TABELLE UNITE + GANTT) ──
for alg in ordine_metodi:
    ws = workbook.add_worksheet(alg)
    kpi = risultati[alg]
    seq = sequenze[alg]
    cmax = kpi['Cmax']
    
    max_cols = max(cmax + 10, 50)
    ws.set_column(0, max_cols, 2.7) 
    
    titolo = f"  DETTAGLIO JOB - {alg} | Cmax={kpi['Cmax']}  Fw_tot={kpi['Fw_tot']}  Tw_tot={kpi['Tw_tot']}  #Tardy={kpi['#Tardy']}"
    ws.merge_range(0, 0, 0, 44, titolo, fmt_tbl_title)
    
    seq_str = " -> ".join(map(str, seq))
    ws.merge_range(1, 0, 1, 44, f"Sequenza: {seq_str}", fmt_tbl_seq)
    ws.set_row(1, 30) 
    
    col_names = ['Job', 'r', 'p', 'd', 'w', 'Ret', 'Mach', 'Setup_ini', 'Setup_fin', 'C', 'F', 'Fw', 'T', 'Tw', 'Ritardo']
    
    for idx, c_name in enumerate(col_names):
        ws.merge_range(2, idx*3, 2, idx*3 + 2, c_name, fmt_tbl_hdr)
    
    for idx, job in enumerate(seq):
        row = idx + 3
        is_tardy = kpi['Uj'][job] == 1
        fmt_row = fmt_tbl_tardy if is_tardy else (fmt_tbl_norm_alt if idx % 2 == 0 else fmt_tbl_norm)
        
        s_ini = kpi['Setup_ini'][job]
        s_fin = kpi['Setup_fin'][job]
        
        data = [
            job, rj[job], pj[job], dj[job], wj[job], reticolo_j[job], 
            kpi['mj'][job], 
            s_ini if s_ini != -1 else "-", 
            s_fin if s_fin != -1 else "-", 
            kpi['Cj'][job], kpi['Fj'][job], kpi['Fwj'][job], 
            kpi['Tj'][job], kpi['Twj'][job],
            "*" if is_tardy else ""
        ]
        
        for c_idx, val in enumerate(data):
            ws.merge_range(row, c_idx*3, row, c_idx*3 + 2, val, fmt_row)

    # GANTT NELLE CELLE (Posizionato subito sotto la tabella principale)
    gantt_start_row = N + 5
    ws.merge_range(gantt_start_row, 0, gantt_start_row, 2, "Mach \\ Timeline", fmt_gantt_mach) 
    
    for t in range(cmax + 1):
        ws.write(gantt_start_row, t + 3, t, fmt_gantt_hdr)
    
    for m in range(M):
        r = gantt_start_row + 1 + m
        ws.merge_range(r, 0, r, 2, f"Macchina {m}", fmt_gantt_mach)
        for t in range(cmax): 
            ws.write_blank(r, t + 3, "", fmt_gantt_grid)
            
    for job in range(N):
        m = kpi['mj'][job]
        r = gantt_start_row + 1 + m
        
        if kpi['Setup_ini'][job] != -1:
            c_start = kpi['Setup_ini'][job] + 3
            c_end   = kpi['Setup_fin'][job] + 2     
            if c_end > c_start:
                if c_end - c_start == 0: 
                    ws.write(r, c_start, "S", fmt_gantt_setup)
                else:
                    ws.merge_range(r, c_start, r, c_end, "S", fmt_gantt_setup)
        
        st  = kpi['STj'][job]
        ct  = kpi['Cj'][job]
        ret = reticolo_j[job]
        
        c_start = st + 3
        c_end   = ct + 2
        
        if c_end >= c_start:
            if c_end == c_start:
                ws.write(r, c_start, job, fmt_gantt_ret[ret])
            else:
                ws.merge_range(r, c_start, r, c_end, job, fmt_gantt_ret[ret])

    # Legenda Reticoli
    leg_row = gantt_start_row + 1 + M + 2
    ws.merge_range(leg_row, 0, leg_row, 2, "Legenda Gantt:", workbook.add_format({'bold': True, 'align': 'right'}))
    ws.write(leg_row, 3, "S", fmt_gantt_setup)
    ws.merge_range(leg_row, 4, leg_row, 8, "Setup R.", workbook.add_format({'align': 'left'}))
    
    for i, col in enumerate(colori_reticoli_hex):
        ws.write(leg_row + 1 + i, 3, "", fmt_gantt_ret[i])
        ws.merge_range(leg_row + 1 + i, 4, leg_row + 1 + i, 8, f"Reticolo {i}", workbook.add_format({'align': 'left'}))

writer.close()
print(f"\nReport completato con successo e salvato in '{excel_filename}'.")