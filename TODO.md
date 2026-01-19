PROGRESS
[██░░░░░░░░░] 15% - bazowy solver i metryki, brak tabel/wykresow

NEXT
🔴 [~90 min] Zrobic generator geometrii pokoju + maski okna/scian i grzejnika z parametrem r (pod eksperyment 1)

MVP
🔴 [~60 min] Rozszerzyc BC o Robin z roznymi lambdami dla okna i scian + u_out (w solverze)
🔴 [~45 min] Ustalic i zapisac zestaw bazowych parametrow (alpha, lambda_okno/sciana, u_out, setpoint, dt, hx) dla exp1
🔴 [~90 min] Zaimplementowac runner exp1: sweep r, symulacja, zapis mu(T) i sigma(T) do CSV
🔴 [~90 min] Wygenerowac tabele i wykresy exp1: sigma(T) vs r, opcjonalnie mu(T) vs r + 2-3 mapy temperatury
🔴 [~90 min] Dodac sterowanie czasowe grzejnikiem (on/off wg przedzialu czasu) pod exp2
🔴 [~90 min] Zdefiniowac maski pomieszczen + scenariusz czasowy (t_out, t_return) dla exp2
🔴 [~120 min] Uruchomic exp2 dla 2 strategii, zapisac mu_i(t), Psi(t), czas powrotu do komfortu; zrobic wykresy i tabele

Po MVP
🟡 [~60 min] Przetestowac exp2 dla 3 temperatur zewnetrznych (bardzo zimno/zimno/chlodno) i dodac wykres porownawczy
🟡 [~60 min] Rozszerzyc exp1 o parametr rozmiaru grzejnika d i zrobic sigma(r,d)
🟡 [~90 min] Analiza bledu numerycznego: zbieznosc po siatce i kroku czasu, wybor stabilnych parametrow
🟡 [~60 min] Opisac model matematyczny i schemat numeryczny w notatniku raportowym
🟡 [~60 min] Ujednolicic estetyke wykresow (etykiety, jednostki, legenda, siatka)
🟢 [~90 min] Zamienic jawny krok na stabilniejszy (np. implicit/ADI) lub dodac automatyczny dobor dt
🟢 [~45 min] Uporzadkowac kod eksperymentow w osobnych skryptach/notebookach
🟢 [~45 min] Dodac krotkie wnioski tekstowe pod kazda tabela/wykres
