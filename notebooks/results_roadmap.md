# Workflow realizacji eksperymentów – ogrzewanie mieszkania

Poniżej opisano krok po kroku, **co należy zaimplementować i w jakiej kolejności**, aby uzyskać wyniki dla dwóch eksperymentów:
1. lokalizacja grzejnika,
2. wyłączanie grzejników podczas nieobecności.

Opis łączy **idee matematyczne i programistyczne w jednym spójnym procesie**.

---

# Eksperyment 1  
## Czy grzejnik musi być pod oknem?

---

## 1. Definicja domeny i geometrii

Rozpatrujemy pojedynczy pokój jako obszar
$$
\Omega \subset \mathbb{R}^2.
$$

W obrębie tej domeny wyróżniamy:
- wnętrze pokoju,
- fragment brzegu odpowiadający oknu,
- fragment brzegu odpowiadający ścianom,
- niewielki obszar wewnętrzny odpowiadający grzejnikowi.

Na poziomie implementacyjnym potrzebny jest mechanizm, który:
- rozpoznaje, do jakiego typu obszaru należy dany punkt,
- potrafi wygenerować obszar grzejnika dla zadanego położenia.

---

## 2. Ewolucja temperatury w czasie

Temperatura w pokoju opisana jest równaniem przewodnictwa ciepła z członem źródłowym:
$$
\frac{\partial u}{\partial t}
=
\alpha \Delta u + f(x, u),
$$

gdzie $u(x,t)$ oznacza temperaturę, a $f(x,u)$ opisuje działanie grzejnika.

Implementacyjnie oznacza to stworzenie procedury, która:
- bierze aktualny rozkład temperatury,
- oblicza nowy rozkład po krótkim kroku czasowym,
- uwzględnia dyfuzję, straty przez brzegi oraz ewentualne grzanie.

---

## 3. Warunki brzegowe (ściany i okno)

Na brzegach pokoju modelujemy ucieczkę ciepła warunkiem typu Robina:
$$
\frac{\partial u}{\partial n}
=
- \frac{\lambda}{\alpha} (u - u_{\text{out}}),
$$

przy czym dla okna i ściany współczynnik $\lambda$ może mieć różne wartości.

W praktyce oznacza to, że:
- punkty przy oknie szybciej tracą ciepło,
- punkty przy ścianach wolniej.

---

## Parametry bazowe (exp1)

Ustalony zestaw parametrów bazowych do uruchamiania exp1:
- $\alpha = 2.1 \times 10^{-5}\ \mathrm{m^2/s}$,
- $\lambda_{\text{sciana}} = 1.0 \times 10^{-4}\ \mathrm{m/s}$,
- $\lambda_{\text{okno}} = 5.0 \times 10^{-4}\ \mathrm{m/s}$,
- $u_{\text{out}} = 0^\circ \mathrm{C}$,
- $S = 20^\circ \mathrm{C}$ (setpoint),
- $h_x = 0.05\ \mathrm{m}$,
- $dt = 0.01\ \mathrm{s}$ (bezpiecznie poniżej limitu stabilności).

---

## 4. Termostat i sterowanie grzejnikiem

Grzejnik działa tylko wtedy, gdy średnia temperatura w pokoju jest niższa od zadanej:
$$
\mu(t) = \frac{1}{|\Omega|} \int_{\Omega} u(x,t)\,dx.
$$

Jeżeli
$$
\mu(t) < S,
$$
to grzejnik dostarcza ciepło, w przeciwnym razie jest wyłączony.

Implementacyjnie oznacza to, że w każdym kroku czasu:
- liczona jest średnia temperatura w pokoju,
- na tej podstawie decyduje się o aktywności źródła ciepła.

---

## 5. Parametryzacja położenia grzejnika

Wprowadzamy parametr $r$, oznaczający odległość grzejnika od okna.

Dla kolejnych wartości $r$:
- generowany jest nowy obszar grzejnika,
- uruchamiana jest pełna symulacja temperatury.

Otrzymujemy w ten sposób serię niezależnych eksperymentów.

---

## 6. Zbieranie wielkości wynikowych

Dla każdej symulacji wyznaczamy:
- średnią temperaturę końcową $\mu(T)$,
- odchylenie standardowe temperatury
$$
\sigma(T)
=
\sqrt{
\frac{1}{|\Omega|}
\int_{\Omega} (u(x,T) - \mu(T))^2\,dx
}.
$$

Dane te są zapisywane w postaci umożliwiającej późniejszą analizę i wizualizację.

---

## 7. Analiza i wizualizacja

Na podstawie zapisanych danych:
- tworzone są mapy temperatury dla wybranych położeń grzejnika,
- rysowany jest wykres $\sigma(T)$ w funkcji $r$,
- opcjonalnie rysowany jest wykres $\mu(T)$ w funkcji $r$.

---

## 8. Podsumowanie wyników

Wyniki zestawiane są w tabeli porównawczej, a następnie interpretowane jakościowo pod kątem komfortu cieplnego.

---

# Eksperyment 2  
## Czy wyłączać grzejniki przed wyjściem z domu?

---

## 1. Definicja mieszkania i scenariusza czasowego

Rozpatrujemy mieszkanie jako obszar
$$
\Omega \subset \mathbb{R}^2
$$
podzielony na pomieszczenia $\Omega_i$.

Definiujemy jeden scenariusz czasowy:
- okres przebywania w domu,
- okres nieobecności,
- powrót i ponowne ogrzewanie.

---

## 2. Równanie ewolucji temperatury

W całym mieszkaniu obowiązuje równanie:
$$
\frac{\partial u}{\partial t}
=
\alpha \Delta u + f(x,t,u),
$$

gdzie człon $f(x,t,u)$ zależy od tego, czy w danym momencie grzejniki są aktywne.

Mechanizm obliczeń temperatury pozostaje ten sam jak w eksperymencie 1.

---

## 3. Strategie sterowania ogrzewaniem

Rozpatrujemy dwie strategie:
- ciągłe ogrzewanie,
- wyłączenie ogrzewania podczas nieobecności.

Matematycznie oznacza to, że funkcja źródła ciepła jest jawnie zależna od czasu:
$$
f(x,t,u) =
\begin{cases}
f_{\text{on}}(x,u), & t \notin (t_{\text{out}}, t_{\text{return}}), \\
0, & t \in (t_{\text{out}}, t_{\text{return}}).
\end{cases}
$$

---

## 4. Wielkości mierzone w eksperymencie

W trakcie symulacji wyznaczane są:
- średnie temperatury w wybranych pomieszczeniach
$$
\mu_i(t) = \frac{1}{|\Omega_i|} \int_{\Omega_i} u(x,t)\,dx,
$$
- całkowita energia dostarczona przez grzejniki
$$
\Psi(T) = \int_0^T \int_{\Omega} f(x,t,u)\,dx\,dt.
$$

Dodatkowo określany jest czas potrzebny na odzyskanie komfortu po powrocie.

---

## 5. Uruchamianie symulacji

Dla każdej strategii:
- symulacja uruchamiana jest z tymi samymi warunkami początkowymi,
- zapisywany jest pełny przebieg temperatury i energii w czasie.

Jedyną różnicą pomiędzy symulacjami jest sposób sterowania grzejnikami.

---

## 6. Porównanie wyników

Na podstawie zapisanych danych:
- porównywane są wykresy temperatury w czasie,
- porównywane są wykresy narastającej energii,
- analizowany jest czas powrotu do komfortu cieplnego.

---

## 7. Wnioski końcowe

Wnioski formułowane są na podstawie porównania:
- całkowitego zużycia energii,
- czasu dyskomfortu,
- zależności wyników od warunków zewnętrznych.

Celem jest odpowiedź na pytanie, czy wyłączanie ogrzewania jest korzystne z punktu widzenia zarówno kosztów, jak i komfortu.
