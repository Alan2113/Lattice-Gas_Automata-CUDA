# LGA — Lattice Gas Automata na GPU

Symulacja przepływu gazu metodą **Lattice Gas Automata** (model HPP, 4 kierunki) zaimplementowana w CUDA z wizualizacją OpenGL w czasie rzeczywistym.

![CUDA](https://img.shields.io/badge/CUDA-supported-76B900?logo=nvidia)
![C++](https://img.shields.io/badge/C++-supported-blue?logo=cplusplus)
![OpenGL](https://img.shields.io/badge/OpenGL-FreeGLUT%20%2B%20GLEW-5586A4?logo=opengl)

## Spis treści

- [Opis modelu](#opis-modelu)
- [Wymagania](#wymagania)
- [Instalacja bibliotek](#instalacja-bibliotek)
- [Budowanie](#budowanie)
- [Uruchomienie](#uruchomienie)
- [Sterowanie](#sterowanie)
- [Konfiguracja](#konfiguracja)
- [Implementacja](#implementacja)

---

## Opis modelu

**Lattice Gas Automata (LGA)** to dyskretny model dynamiki płynów. Przestrzeń jest podzielona na siatkę komórek, w których "krążą" wirtualne cząstki o ustalonych kierunkach prędkości. Ewolucja systemu polega na cyklicznym powtarzaniu dwóch kroków:

1. **Collision** — w każdej komórce cząstki zderzają się według prostych reguł zachowujących masę i pęd
2. **Streaming** — po kolizji cząstki przemieszczają się do sąsiednich komórek zgodnie ze swoim kierunkiem

W skali makroskopowej z lokalnych reguł emerguje zachowanie zgodne z równaniami Naviera-Stokesa.

### Wariant: HPP (4 kierunki)

W tej implementacji każda komórka przechowuje 4 bity reprezentujące cząstki poruszające się w kierunkach: **prawo, góra, lewo, dół**. Reguła kolizji: gdy w komórce spotykają się dwie cząstki przeciwbieżne (poziomo lub pionowo), zmieniają się w parę poprzeczną.

### Scenariusz symulacji

- Siatka **128×128** komórek otoczona ścianami
- Pionowa **bariera w środku** z 16-komórkową przerwą
- Cząstki inicjowane po **lewej stronie bariery** z gęstością ~8%
- Obserwujemy przepływ przez przerwę i tworzące się wiry

## Wymagania

| Komponent | Wymaganie |
|-----------|--------|
| **CUDA Toolkit** | wymagany (z nvcc) |
| **CMake** | 3.24+ |
| **Kompilator C++** | MSVC 2019/2022 (Visual Studio Build Tools) |
| **Karta graficzna** | NVIDIA z Compute Capability ≥ 7.5 |
| **System** | Windows x64 |

> **Uwaga:** Architektura CUDA w `CMakeLists.txt` ustawiona jest na `89` (RTX 4070, Ada Lovelace). Dla innych kart zmień wartość — np. `86` dla RTX 30xx, `75` dla RTX 20xx.

## Instalacja bibliotek

Repozytorium nie zawiera bibliotek graficznych — pobierz je ręcznie do folderu `libs/`.

### freeglut 3.4.0
Pobierz wersję MSVC binary z https://www.transmissionzero.co.uk/software/freeglut-devel/ i rozpakuj jako `libs/freeglut/`.

### GLEW 2.2.0
Pobierz Windows binary z https://glew.sourceforge.net/ i rozpakuj jako `libs/glew-2.2.0/`.

### Wynikowa struktura

```
1.LGA/
├── CMakeLists.txt
├── main.cu
├── libs/
│   ├── freeglut/
│   │   ├── include/GL/
│   │   ├── lib/x64/freeglut.lib
│   │   └── bin/x64/freeglut.dll
│   └── glew-2.2.0/
│       ├── include/GL/
│       ├── lib/Release/x64/glew32.lib
│       └── bin/Release/x64/glew32.dll
└── README.md
```

## Budowanie

### Z CLion

1. **File → Open** → wskaż folder `1.LGA`
2. **Settings → Build → CMake** — ustaw toolchain na **Visual Studio**
3. **Tools → CMake → Reset Cache and Reload Project**
4. Wybierz target **LGA** z dropdown'a obok przycisku Run i zbuduj (Ctrl+F9)

### Z linii poleceń (PowerShell / cmd)

```cmd
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --target LGA
```

## Uruchomienie

```cmd
build\LGA.exe
```

CMake automatycznie kopiuje wymagane pliki DLL (`freeglut.dll`, `glew32.dll`) obok pliku wykonywalnego, więc aplikacja uruchamia się od razu bez ręcznego kopiowania.

Po uruchomieniu w konsoli pojawi się informacja o wykrytej karcie GPU:

```
Karta: NVIDIA GeForce RTX 4070 Laptop GPU
Compute Capability: 8.9
Rdzenie CUDA: 36
Pamiec: 8188 MB
```

## Sterowanie

| Klawisz | Akcja |
|---------|-------|
| `Spacja` | Pauza / wznowienie |
| `+` lub `=` | Przyspieszenie symulacji |
| `-` lub `_` | Zwolnienie symulacji |
| `R` | Reset symulacji |
| `Esc` | Wyjście |

## Konfiguracja

Parametry modyfikowalne na początku pliku `main.cu`:

```cpp
const int Nx = 128, Ny = 128;       // rozmiar siatki
const int WINDOW_SIZE = 800;        // rozmiar okna w pikselach
const float PARTICLE_PROB = 0.08f;  // gęstość cząstek (0.0–1.0)
```

Po zmianie parametrów wymagany jest rebuild.

## Implementacja

### Kernele CUDA

- **`collisionAndStreaming`** — łączy etapy kolizji i streamingu w jednym przebiegu, z odbiciem od ścian. Używa `atomicAdd` przy zapisie do sąsiednich komórek dla bezpiecznej współbieżności.
- **`computeDensity`** — oblicza gęstość cząstek (suma 4 kierunków) dla wizualizacji.
- **`clearGrid`** — zeruje siatkę przed kolejnym krokiem streamingu.

### Layout pamięci

Stan komórki przechowywany jako **`int4`** (4 × 32 bit) odpowiadający 4 kierunkom prędkości — dzięki temu odczyt/zapis komórki to jedna instrukcja pamięciowa GPU. Używane są dwa bufory (`d_current`, `d_next`) zamieniane wskaźnikami po każdym kroku.

### Wizualizacja

Każda komórka rysowana jest jako punkt OpenGL (`GL_POINTS` o rozmiarze 6 px). Kolor zależy od stanu komórki — białe punkty oznaczają cząstki, ciemnoniebieskie ściany.
