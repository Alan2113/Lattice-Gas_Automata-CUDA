#include <GL/glew.h>
#include <GL/freeglut.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Parametry
const int Nx = 128, Ny = 128;
const int WINDOW_SIZE = 800;
const float PARTICLE_PROB = 0.08f;

// GPU: 2 tablice 4-kanałowe zamiast 8 osobnych
int4 *d_current, *d_next;  // int4 = {x, y, z, w} = {prawo, góra, lewo, dół}
int *d_walls;

// CPU
int *h_density, *h_walls;

// Stan
int step = 0;
bool paused = false;
int speed_factor = 5;

// ==== KERNELE ====
__global__ void computeDensity(int4 *cells, int *density, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = x + y * Nx;
    int4 c = cells[idx];
    density[idx] = c.x + c.y + c.z + c.w;
}

__global__ void collisionAndStreaming(int4 *curr, int4 *next, int *walls, int Nx, int Ny) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= Nx || y >= Ny) return;

    int idx = x + y * Nx;

    // Ściany
    if (walls[idx]) {
        next[idx] = make_int4(0, 0, 0, 0);
        return;
    }

    int4 in = curr[idx];
    int4 out = in;  // domyślnie bez zmian

    // === COLLISION ===
    if (in.x == 1 && in.y == 0 && in.z == 1 && in.w == 0) {
        out = make_int4(0, 1, 0, 1);  // ←→ → ↑↓
    } else if (in.x == 0 && in.y == 1 && in.z == 0 && in.w == 1) {
        out = make_int4(1, 0, 1, 0);  // ↑↓ → ←→
    }

    // === STREAMING - zapisz do sąsiadów ===
    // Prawo (x): idzie do (x+1)
    if (x < Nx - 1 && !walls[idx + 1])
        atomicAdd(&next[idx + 1].x, out.x);
    else
        atomicAdd(&next[idx].z, out.x);  // odbicie

    // Góra (y): idzie do (y+1)
    if (y < Ny - 1 && !walls[idx + Nx])
        atomicAdd(&next[idx + Nx].y, out.y);
    else
        atomicAdd(&next[idx].w, out.y);  // odbicie

    // Lewo (z): idzie do (x-1)
    if (x > 0 && !walls[idx - 1])
        atomicAdd(&next[idx - 1].z, out.z);
    else
        atomicAdd(&next[idx].x, out.z);  // odbicie

    // Dół (w): idzie do (y-1)
    if (y > 0 && !walls[idx - Nx])
        atomicAdd(&next[idx - Nx].w, out.w);
    else
        atomicAdd(&next[idx].y, out.w);  // odbicie
}

__global__ void clearGrid(int4 *grid, int Nx, int Ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < Nx * Ny) grid[idx] = make_int4(0, 0, 0, 0);
}

// ==== INICJALIZACJA ====
void initLGA() {
    size_t size = Nx * Ny * sizeof(int4);

    cudaMalloc(&d_current, size);
    cudaMalloc(&d_next, size);
    cudaMalloc(&d_walls, Nx * Ny * sizeof(int));

    h_density = (int*)malloc(Nx * Ny * sizeof(int));
    h_walls = (int*)calloc(Nx * Ny, sizeof(int));

    // Ramka
    for (int x = 0; x < Nx; x++) {
        h_walls[x] = h_walls[x + (Ny-1)*Nx] = 1;
    }
    for (int y = 0; y < Ny; y++) {
        h_walls[y*Nx] = h_walls[Nx-1 + y*Nx] = 1;
    }

    // Bariera z przerwą
    int barrier_x = Nx / 2;
    for (int y = 0; y < Ny; y++) {
        if (y < Ny/2 - 8 || y > Ny/2 + 8) {
            h_walls[barrier_x + y * Nx] = 1;
        }
    }

    cudaMemcpy(d_walls, h_walls, Nx*Ny*sizeof(int), cudaMemcpyHostToDevice);

    // Cząstki po lewej
    int4 *h_cells = (int4*)calloc(Nx * Ny, sizeof(int4));
    srand(time(NULL));

    for (int y = 10; y < Ny - 10; y++) {
        for (int x = 10; x < barrier_x - 5; x++) {
            int idx = x + y * Nx;
            if (h_walls[idx]) continue;

            h_cells[idx].x = (rand() % 100 < PARTICLE_PROB * 100) ? 1 : 0;
            h_cells[idx].y = (rand() % 100 < PARTICLE_PROB * 100) ? 1 : 0;
            h_cells[idx].z = (rand() % 100 < PARTICLE_PROB * 100) ? 1 : 0;
            h_cells[idx].w = (rand() % 100 < PARTICLE_PROB * 100) ? 1 : 0;
        }
    }

    cudaMemcpy(d_current, h_cells, size, cudaMemcpyHostToDevice);
    free(h_cells);

    printf("LGA: %dx%d | Bariera: x=%d | Prob: %.2f%%\n",
           Nx, Ny, barrier_x, PARTICLE_PROB*100);
}

// ==== SYMULACJA ====
void simulate() {
    if (paused) return;

    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    // Wyczyść next
    clearGrid<<<(Nx*Ny + 255)/256, 256>>>(d_next, Nx, Ny);

    // Collision + Streaming w jednym kernelu
    collisionAndStreaming<<<grid, block>>>(d_current, d_next, d_walls, Nx, Ny);

    // Swap
    int4 *temp = d_current;
    d_current = d_next;
    d_next = temp;

    step++;
}

// ==== WIZUALIZACJA ====
void display() {
    glClear(GL_COLOR_BUFFER_BIT);

    // Pobierz gęstość
    dim3 block(16, 16);
    dim3 grid((Nx + 15) / 16, (Ny + 15) / 16);

    int *d_density;
    cudaMalloc(&d_density, Nx * Ny * sizeof(int));
    computeDensity<<<grid, block>>>(d_current, d_density, Nx, Ny);
    cudaMemcpy(h_density, d_density, Nx*Ny*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_density);

    // Rysuj
    glPointSize(6.0f);
    glBegin(GL_POINTS);
    for (int y = 0; y < Ny; y++) {
        for (int x = 0; x < Nx; x++) {
            int idx = x + y * Nx;

            if (h_walls[idx]) {
                glColor3f(0.0f, 0.1f, 0.3f); //BOKI
                glVertex2f(x + 0.5f, y + 0.5f);
            } else if (h_density[idx] > 0) {
                float i = h_density[idx] / 4.0f;
                glColor3f(1.0f, 1.0f, 1.0f); //kwadraty
                //glColor3f(0.3f + i*0.7f, 0.5f + i*0.5f, 1.0f); //kwadraty
                glVertex2f(x + 0.5f, y + 0.5f);
            }
        }
    }
    glEnd();

    // Info
    char info[100];
    sprintf(info, "Krok:%d %s v:%d", step, paused?"[||]":"", speed_factor);
    glColor3f(0.0f, 0.7f, 0.1f);
    glRasterPos2f(5, Ny - 5);
    for (char *c = info; *c; c++)
        glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, *c);

    glutSwapBuffers();
}

void idle() {
    static int skip = 0;
    if (++skip < speed_factor) { glutPostRedisplay(); return; }
    skip = 0;

    simulate();
    glutPostRedisplay();

    if (step % 100 == 0 && !paused) printf("Krok: %d\n", step);
}

void keyboard(unsigned char key, int x, int y) {
    switch(key) {
        case ' ': paused = !paused; break;
        case '+': case '=': if (speed_factor > 1) speed_factor--; break;
        case '-': case '_': if (speed_factor < 10) speed_factor++; break;
        case 'c': case 'C': {
            int4 *empty = (int4*)calloc(Nx * Ny, sizeof(int4));
            cudaMemcpy(d_current, empty, Nx*Ny*sizeof(int4), cudaMemcpyHostToDevice);
            free(empty);
            break;
        }
        case 'r': case 'R':
            cudaFree(d_current); cudaFree(d_next); cudaFree(d_walls);
            free(h_density); free(h_walls);
            step = 0; paused = false;
            initLGA();
            break;
        case 27: exit(0);
    }
}

void reshape(int w, int h) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D(0, Nx, 0, Ny);
    glMatrixMode(GL_MODELVIEW);
}

//additional
void mouse(int button, int state, int x, int y) {
    if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
        // Konwertuj pozycję myszy na koordynaty siatki
        int gx = x * Nx / WINDOW_SIZE;
        int gy = (WINDOW_SIZE - y) * Ny / WINDOW_SIZE;  // y odwrócone

        // Dodaj cząstki w promieniu 5 wokół klikniętego punktu
        int4 *temp = (int4*)malloc(Nx * Ny * sizeof(int4));
        cudaMemcpy(temp, d_current, Nx*Ny*sizeof(int4), cudaMemcpyDeviceToHost);

        for (int dy = -5; dy <= 5; dy++) {
            for (int dx = -5; dx <= 5; dx++) {
                int nx = gx + dx, ny = gy + dy;
                if (nx >= 0 && nx < Nx && ny >= 0 && ny < Ny) {
                    int idx = nx + ny * Nx;
                    if (!h_walls[idx]) {
                        temp[idx] = make_int4(1, 1, 1, 1);
                    }
                }
            }
        }

        cudaMemcpy(d_current, temp, Nx*Ny*sizeof(int4), cudaMemcpyHostToDevice);
        free(temp);
    }
}


int main(int argc, char** argv) {

    // === WYKRYJ KARTĘ GPU ===
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount > 0) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        printf("\ninfo o gpu\n");
        printf("Karta: %s\n", prop.name);
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("Rdzenie CUDA: %d\n", prop.multiProcessorCount);
        printf("Pamiec: %.0f MB\n", prop.totalGlobalMem / 1024.0 / 1024.0);
    } else {
        printf("jakis blad\n");
        return 1;
    }





    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(WINDOW_SIZE, WINDOW_SIZE);
    glutCreateWindow("LGA - Optimized");
    glewInit();

    initLGA();

    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutIdleFunc(idle);
    glutKeyboardFunc(keyboard);
    glutMouseFunc(mouse);
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);

    printf("SPACJA=pauza | +/-=predkosc | R=reset | ESC=wyjscie\n");
    glutMainLoop();


    return 0;
}
