1) Vybereme Dataset
2) Uděláme funkci kterou chceme testovat dané modely (linearni regrese / klasifikace / ANN)
3) Pomocí 10-fold cross validace (dále CV) rozdělíme dataset na desetiny přičemž vždy jedna desetina je test set
4) Použijeme regresi/klasifikiaci 10x na všechny možné varianty CV
5) vybereme ten model s nejmenším errorem
6) pokud děláme two-layer CV přidíme ještě jeden for, ve kterém vezmeme nejlepší odel z inner loopu, a otestujeme ho na novém k-fold datasetu (který je rozdělený jinak než v inner loopu)

