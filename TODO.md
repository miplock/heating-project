# TODO - heating project

## Wybór eksperymentow
- [ ] Zdefiniowac scenariusze i geometrie (uklad pomieszczen, okna, grzejniki)
- [ ] Okreslic temperature zewnetrzna dla kazdego eksperymentu

## Dane fizyczne
- [ ] Zebrac stale fizyczne (rho, c, alpha, moce grzejnikow itp.)
- [ ] Zapisac zrodla danych i wrzucic je do `data/` (csv/json)

## Model i numeryka
- [ ] Zdefiniowac model PDE (rownanie ciepla + grzanie), warunki brzegowe
- [ ] Wybrac schemat dyskretyzacji (zalecany: niejawny FDM)
- [ ] Zrobic analize bledu / indykatora bledu i dobrac kroki hx, ht
- [ ] Zadbaj o wektoryzacje obliczen (unikac zagniezdzonych petli)

## Implementacja
- [ ] Wydzielic `notebooks/` (albo potwierdzic, ze notatniki zostaja w `pipeline/`)

## Wyniki i wizualizacje
- [ ] Przygotowac wykresy i animacje temperatury
- [ ] Dla eksperymentu 1: analiza sigma_u(r, d) i sredniej temperatury
- [ ] Dla eksperymentu 2: porownanie energii Psi(T) dla strategii grzania
- [ ] Dla eksperymentu 3: porownanie strategii ciagle grzanie vs ochlodzenie+dogrzanie

## Raport / aplikacja
- [ ] Zdecydowac: raport PDF albo aplikacja Streamlit
- [ ] Raport: opis praktyczny, opis matematyczny, analiza bledu, wyniki, wnioski
- [ ] Raport: tabela stalych fizycznych + zrodla

## Repo
- [ ] `readme.md` (opis projektu, struktura, uruchomienie)
- [ ] `requirements.txt`
- [ ] Upewnic sie, ze `data/` i `notebooks/` sa w repo (jesli beda uzywane)
- [ ] .gitignore (gotowe)

## Metadane
- [ ] Dodac krotka notke o uzyciu AI (jesli uzywana)
