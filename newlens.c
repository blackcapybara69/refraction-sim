// NewLens | LightRay Optics  -  refraction simulator (raylib)
// Build example:  gcc refraction.c -o newlens -lraylib -lm   (add -lopengl32 -lgdi32 -lwinmm on Windows)

#include "raylib.h"
#include "raymath.h"
#include <math.h>

#define MAX_OBJECTS 32
#define MAX_VERTS   64
#define MAX_RAYS    512
#define MAX_SEGS    8000
#define MAX_DEPTH   10
#define MAX_INFO    6
#define MAX_WAVES   24

#define EMITTER_W   100.0f
#define EMITTER_H   70.0f
#define HANDLE_DIST 95.0f

typedef enum { SHAPE_SLAB, SHAPE_PRISM, SHAPE_LENS, SHAPE_HALFDISK, SHAPE_MIRROR } ShapeType;

typedef struct {
    ShapeType type;
    Vector2 pos;
    float rot;      // degrees
    float scale;
    float n;        // refractive index (at 589 nm)
    bool mirror;
    int count;
    Vector2 local[MAX_VERTS];
    Vector2 world[MAX_VERTS];
    float orient;   // +1 / -1 polygon winding
} Object;

typedef struct { Vector2 pos; float angle; } Emitter;

typedef struct {
    Vector2 o, d;
    float I;        // relative intensity (used for culling)
    float wl;       // wavelength in nm
    float bright;   // constant brightness multiplier of this ray
    int depth;
    bool info;      // follows the "measured" ray for angle read-outs
} LightRay;

typedef struct { Vector2 a, b; float wl; float alpha; } Seg;

typedef struct {
    Vector2 p, n;
    float thetaI, thetaT;
    float n1, n2;
    bool tir, mirror;
} Hit;

// ---------------------------------------------------------------- state
static Object objs[MAX_OBJECTS];
static int objCount = 0;
static int sel = -1;

static Emitter emitter = { {170, 350}, 0.0f };

static Seg segs[MAX_SEGS];
static int segCount = 0;
static Hit infos[MAX_INFO];
static int infoCount = 0;

static const char *colorNames[] = { "Red", "Green", "Blue", "Violet", "White (dispersion)" };
static const float colorWl[]    = { 650, 532, 460, 410, 0 };
static int laserColor = 4;

static const int beamOptions[] = { 1, 3, 5, 9 };
static int beamIdx = 0;

static float dispersion = 6.0f;   // exaggerated on purpose so rainbows are visible
static bool showNormals = true;
static bool useFresnel  = true;
static bool showGrid    = true;
static bool showHelp    = true;

static const float presets[] = { 1.33f, 1.45f, 1.52f, 1.62f, 2.42f };

// ---------------------------------------------------------------- helpers
static float Cross2(Vector2 a, Vector2 b) { return a.x * b.y - a.y * b.x; }

static Color WithAlpha(Color c, float a)
{
    c.a = (unsigned char)(Clamp(a, 0.0f, 1.0f) * 255.0f);
    return c;
}

static Color WavelengthToColor(float wl)
{
    float r = 0, g = 0, b = 0, f = 1;
    if (wl >= 380 && wl < 440)      { r = -(wl - 440) / 60.0f; b = 1; }
    else if (wl >= 440 && wl < 490) { g = (wl - 440) / 50.0f; b = 1; }
    else if (wl >= 490 && wl < 510) { g = 1; b = -(wl - 510) / 20.0f; }
    else if (wl >= 510 && wl < 580) { r = (wl - 510) / 70.0f; g = 1; }
    else if (wl >= 580 && wl < 645) { r = 1; g = -(wl - 645) / 65.0f; }
    else if (wl >= 645 && wl <= 780){ r = 1; }
    if (wl < 420)      f = 0.3f + 0.7f * (wl - 380) / 40.0f;
    else if (wl > 700) f = 0.3f + 0.7f * (780 - wl) / 80.0f;
    return (Color){
        (unsigned char)(powf(r * f, 0.8f) * 255),
        (unsigned char)(powf(g * f, 0.8f) * 255),
        (unsigned char)(powf(b * f, 0.8f) * 255), 255 };
}

// Cauchy-style dispersion: shorter wavelengths bend more
static float IndexAt(const Object *o, float wl)
{
    float um = wl / 1000.0f;
    float B = 0.0042f * dispersion * (o->n - 1.0f) / 0.5f;
    return o->n + B * (1.0f / (um * um) - 1.0f / (0.589f * 0.589f));
}

// ---------------------------------------------------------------- shapes
static void AddV(Object *o, float x, float y)
{
    if (o->count < MAX_VERTS) o->local[o->count++] = (Vector2){ x, y };
}

static void BuildShape(Object *o, ShapeType t)
{
    o->type = t;
    o->count = 0;
    o->mirror = false;
    o->scale = 1.0f;
    o->rot = 0.0f;
    o->n = 1.5f;

    switch (t)
    {
    case SHAPE_SLAB:
        AddV(o, -120, -60); AddV(o, 120, -60); AddV(o, 120, 60); AddV(o, -120, 60);
        break;
    case SHAPE_PRISM:
        AddV(o, 0, -104); AddV(o, 90, 52); AddV(o, -90, 52);
        break;
    case SHAPE_LENS:
    {
        const float R = 180.0f, h = 100.0f, off = sqrtf(R * R - h * h);
        const int N = 16;
        for (int k = 0; k <= N; k++)                 // right surface, top -> bottom
        {
            float y = -h + 2.0f * h * k / N;
            AddV(o, sqrtf(R * R - y * y) - off, y);
        }
        for (int k = N - 1; k >= 1; k--)             // left surface, bottom -> top
        {
            float y = -h + 2.0f * h * k / N;
            AddV(o, -(sqrtf(R * R - y * y) - off), y);
        }
        break;
    }
    case SHAPE_HALFDISK:
    {
        const float R = 110.0f;
        const int N = 24;
        for (int k = 0; k <= N; k++)
        {
            float a = (-90.0f + 180.0f * k / N) * DEG2RAD;
            AddV(o, R * cosf(a) - 45.0f, R * sinf(a));
        }
        break;
    }
    case SHAPE_MIRROR:
        AddV(o, -110, -5); AddV(o, 110, -5); AddV(o, 110, 5); AddV(o, -110, 5);
        o->mirror = true;
        break;
    }
}

static void UpdateWorld(Object *o)
{
    float rad = o->rot * DEG2RAD;
    float area = 0.0f;
    for (int i = 0; i < o->count; i++)
        o->world[i] = Vector2Add(Vector2Rotate(Vector2Scale(o->local[i], o->scale), rad), o->pos);
    for (int i = 0; i < o->count; i++)
    {
        Vector2 a = o->world[i], b = o->world[(i + 1) % o->count];
        area += Cross2(a, b);
    }
    o->orient = (area >= 0.0f) ? 1.0f : -1.0f;
}

static bool PointInObject(const Object *o, Vector2 p)
{
    for (int i = 0; i < o->count; i++)
    {
        Vector2 a = o->world[i], b = o->world[(i + 1) % o->count];
        if (Cross2(Vector2Subtract(b, a), Vector2Subtract(p, a)) * o->orient < 0.0f) return false;
    }
    return true;
}

static int PickObject(Vector2 p)
{
    for (int i = objCount - 1; i >= 0; i--)
        if (PointInObject(&objs[i], p)) return i;
    return -1;
}

static void SpawnObject(ShapeType t, Vector2 at)
{
    if (objCount >= MAX_OBJECTS) return;
    Object *o = &objs[objCount];
    BuildShape(o, t);
    o->pos = at;
    UpdateWorld(o);
    sel = objCount++;
}

static void RemoveObject(int i)
{
    if (i < 0 || i >= objCount) return;
    for (int k = i; k < objCount - 1; k++) objs[k] = objs[k + 1];
    objCount--;
    sel = -1;
}

// ---------------------------------------------------------------- ray tracing
static void PushRay(LightRay *stack, int *sp, LightRay r)
{
    if (*sp < MAX_RAYS) stack[(*sp)++] = r;
}

static void AddSeg(Vector2 a, Vector2 b, float wl, float alpha)
{
    if (segCount < MAX_SEGS) segs[segCount++] = (Seg){ a, b, wl, alpha };
}

static void Trace(void)
{
    static LightRay stack[MAX_RAYS];
    int sp = 0;
    segCount = 0;
    infoCount = 0;

    Vector2 dir = { cosf(emitter.angle * DEG2RAD), sinf(emitter.angle * DEG2RAD) };
    Vector2 perp = { -dir.y, dir.x };
    Vector2 tip = Vector2Add(emitter.pos, Vector2Scale(dir, EMITTER_W * 0.5f));

    int nb = beamOptions[beamIdx];
    bool white = (colorWl[laserColor] == 0.0f);
    int nw = white ? MAX_WAVES : 1;
    float spread = 30.0f;

    for (int k = 0; k < nb; k++)
    {
        float off = (nb > 1) ? ((float)k / (nb - 1) * 2.0f - 1.0f) * spread : 0.0f;
        Vector2 o = Vector2Add(tip, Vector2Scale(perp, off));
        for (int w = 0; w < nw; w++)
        {
            LightRay r;
            r.o = o;
            r.d = dir;
            r.I = 1.0f;
            r.wl = white ? 400.0f + 300.0f * w / (nw - 1) : colorWl[laserColor];
            r.bright = white ? 4.0f / nw : 1.0f;
            r.depth = 0;
            r.info = (k == nb / 2) && (w == nw / 2);
            PushRay(stack, &sp, r);
        }
    }

    while (sp > 0)
    {
        LightRay r = stack[--sp];

        float bestT = 1e9f;
        int bo = -1, be = -1;
        for (int oi = 0; oi < objCount; oi++)
        {
            Object *ob = &objs[oi];
            for (int ei = 0; ei < ob->count; ei++)
            {
                Vector2 a = ob->world[ei], b = ob->world[(ei + 1) % ob->count];
                Vector2 e = Vector2Subtract(b, a);
                float den = Cross2(r.d, e);
                if (fabsf(den) < 1e-8f) continue;
                Vector2 ao = Vector2Subtract(a, r.o);
                float t = Cross2(ao, e) / den;
                float u = Cross2(ao, r.d) / den;
                if (t > 0.01f && u >= 0.0f && u <= 1.0f && t < bestT) { bestT = t; bo = oi; be = ei; }
            }
        }

        float tEnd = (bo >= 0) ? bestT : 4000.0f;
        Vector2 P = Vector2Add(r.o, Vector2Scale(r.d, tEnd));
        AddSeg(r.o, P, r.wl, r.I * r.bright);

        if (bo < 0 || r.depth >= MAX_DEPTH) continue;

        Object *ob = &objs[bo];
        Vector2 a = ob->world[be], b = ob->world[(be + 1) % ob->count];
        Vector2 e = Vector2Subtract(b, a);
        Vector2 N = Vector2Normalize((Vector2){ e.y * ob->orient, -e.x * ob->orient }); // outward
        bool exiting = Vector2DotProduct(r.d, N) > 0.0f;
        if (exiting) N = Vector2Negate(N);   // N now faces the incoming ray

        float cosi = Clamp(-Vector2DotProduct(N, r.d), 0.0f, 1.0f);
        Vector2 refl = Vector2Normalize(Vector2Subtract(r.d, Vector2Scale(N, 2.0f * Vector2DotProduct(r.d, N))));

        Hit h = { 0 };
        h.p = P;
        h.n = N;
        h.thetaI = acosf(cosi) * RAD2DEG;

        if (ob->mirror)
        {
            h.mirror = true;
            h.thetaT = h.thetaI;
            h.n1 = h.n2 = 1.0f;
            if (r.info && infoCount < MAX_INFO) infos[infoCount++] = h;
            LightRay c = r;
            c.o = Vector2Add(P, Vector2Scale(refl, 0.05f));
            c.d = refl;
            c.I *= 0.95f;
            c.depth++;
            if (c.I > 0.02f) PushRay(stack, &sp, c);
            continue;
        }

        float nObj = IndexAt(ob, r.wl);
        float n1 = exiting ? nObj : 1.0f;
        float n2 = exiting ? 1.0f : nObj;
        float eta = n1 / n2;
        float sin2t = eta * eta * (1.0f - cosi * cosi);
        h.n1 = n1;
        h.n2 = n2;

        if (sin2t >= 1.0f)
        {
            // total internal reflection
            h.tir = true;
            if (r.info && infoCount < MAX_INFO) infos[infoCount++] = h;
            LightRay c = r;
            c.o = Vector2Add(P, Vector2Scale(refl, 0.05f));
            c.d = refl;
            c.depth++;
            PushRay(stack, &sp, c);
            continue;
        }

        float cost = sqrtf(1.0f - sin2t);
        h.thetaT = acosf(cost) * RAD2DEG;
        Vector2 T = Vector2Normalize(Vector2Add(Vector2Scale(r.d, eta), Vector2Scale(N, eta * cosi - cost)));

        float R = 0.0f;
        if (useFresnel)
        {
            float rs = (n1 * cosi - n2 * cost) / (n1 * cosi + n2 * cost);
            float rp = (n1 * cost - n2 * cosi) / (n1 * cost + n2 * cosi);
            R = 0.5f * (rs * rs + rp * rp);
        }

        if (r.info && infoCount < MAX_INFO) infos[infoCount++] = h;

        LightRay t = r;
        t.o = Vector2Add(P, Vector2Scale(T, 0.05f));
        t.d = T;
        t.I = r.I * (1.0f - R);
        t.depth++;
        if (t.I > 0.02f) PushRay(stack, &sp, t);

        if (useFresnel && r.I * R > 0.02f)
        {
            LightRay f = r;
            f.o = Vector2Add(P, Vector2Scale(refl, 0.05f));
            f.d = refl;
            f.I = r.I * R;
            f.depth++;
            f.info = false;
            PushRay(stack, &sp, f);
        }
    }
}

// ---------------------------------------------------------------- drawing
static void FillTri(Vector2 a, Vector2 b, Vector2 c, Color col)
{
    if (Cross2(Vector2Subtract(b, a), Vector2Subtract(c, a)) > 0.0f) DrawTriangle(a, c, b, col);
    else DrawTriangle(a, b, c, col);
}

static void DrawObject(const Object *o, bool selected)
{
    Vector2 c = { 0, 0 };
    for (int i = 0; i < o->count; i++) c = Vector2Add(c, o->world[i]);
    c = Vector2Scale(c, 1.0f / o->count);

    Color fill;
    if (o->mirror) fill = (Color){ 205, 210, 225, 255 };
    else fill = (Color){ 130, 190, 255, (unsigned char)Clamp(30 + (o->n - 1.0f) * 55.0f, 30, 110) };

    for (int i = 0; i < o->count; i++)
        FillTri(c, o->world[i], o->world[(i + 1) % o->count], fill);

    Color edge = o->mirror ? (Color){ 245, 248, 255, 255 } : (Color){ 190, 225, 255, 190 };
    if (selected) edge = (Color){ 255, 235, 120, 255 };
    for (int i = 0; i < o->count; i++)
        DrawLineEx(o->world[i], o->world[(i + 1) % o->count], selected ? 3.0f : 2.0f, edge);
}

static void DrawRays(void)
{
    BeginBlendMode(BLEND_ADDITIVE);
    for (int i = 0; i < segCount; i++)
    {
        Seg *s = &segs[i];
        Color c = WavelengthToColor(s->wl);
        Color core = { (unsigned char)(c.r + (255 - c.r) * 0.35f),
                       (unsigned char)(c.g + (255 - c.g) * 0.35f),
                       (unsigned char)(c.b + (255 - c.b) * 0.35f), 255 };
        DrawLineEx(s->a, s->b, 14.0f, WithAlpha(c, 0.07f * s->alpha));
        DrawLineEx(s->a, s->b,  7.0f, WithAlpha(c, 0.18f * s->alpha));
        DrawLineEx(s->a, s->b,  3.0f, WithAlpha(core, 0.85f * s->alpha));
    }
    EndBlendMode();
}

static void DrawDashedLine(Vector2 a, Vector2 b, float dash, Color col)
{
    Vector2 d = Vector2Subtract(b, a);
    float len = Vector2Length(d);
    if (len < 1.0f) return;
    d = Vector2Scale(d, 1.0f / len);
    for (float t = 0; t < len; t += dash * 2.0f)
        DrawLineEx(Vector2Add(a, Vector2Scale(d, t)),
                   Vector2Add(a, Vector2Scale(d, fminf(t + dash, len))), 1.5f, col);
}

static void DrawNormals(void)
{
    for (int i = 0; i < infoCount; i++)
    {
        Hit *h = &infos[i];
        DrawDashedLine(Vector2Subtract(h->p, Vector2Scale(h->n, 45)),
                       Vector2Add(h->p, Vector2Scale(h->n, 45)), 4.0f, (Color){ 255, 255, 255, 110 });
        DrawCircleV(h->p, 3.0f, WHITE);

        const char *txt;
        if (h->mirror)     txt = TextFormat("mirror  i=r=%.1f deg", h->thetaI);
        else if (h->tir)   txt = TextFormat("i=%.1f deg  TOTAL INTERNAL REFLECTION", h->thetaI);
        else               txt = TextFormat("i=%.1f  r=%.1f deg   n %.2f -> %.2f", h->thetaI, h->thetaT, h->n1, h->n2);

        int w = MeasureText(txt, 14);
        Vector2 lp = { h->p.x + 14, h->p.y - 24 - i * 2 };
        DrawRectangle((int)lp.x - 4, (int)lp.y - 2, w + 8, 18, (Color){ 0, 0, 0, 150 });
        DrawText(txt, (int)lp.x, (int)lp.y, 14, (Color){ 255, 255, 255, 235 });
    }
}

static void DrawEmitter(void)
{
    float rad = emitter.angle * DEG2RAD;
    Vector2 hw = { EMITTER_W * 0.5f, EMITTER_H * 0.5f };
    Vector2 c[4] = { { -hw.x, -hw.y }, { hw.x, -hw.y }, { hw.x, hw.y }, { -hw.x, hw.y } };
    for (int i = 0; i < 4; i++) c[i] = Vector2Add(Vector2Rotate(c[i], rad), emitter.pos);

    Color body = { 85, 90, 108, 255 };
    FillTri(c[0], c[1], c[2], body);
    FillTri(c[0], c[2], c[3], body);
    for (int i = 0; i < 4; i++) DrawLineEx(c[i], c[(i + 1) % 4], 2.0f, (Color){ 170, 176, 200, 255 });

    // front aperture
    Color lc = (colorWl[laserColor] == 0.0f) ? WHITE : WavelengthToColor(colorWl[laserColor]);
    Vector2 dir = { cosf(rad), sinf(rad) };
    Vector2 tip = Vector2Add(emitter.pos, Vector2Scale(dir, EMITTER_W * 0.5f));
    DrawCircleV(tip, 5.0f, lc);

    // rotate handle
    Vector2 hp = Vector2Add(emitter.pos, Vector2Scale(dir, HANDLE_DIST));
    DrawCircleLines((int)hp.x, (int)hp.y, 9.0f, (Color){ 255, 255, 255, 150 });
    DrawCircleV(hp, 3.0f, (Color){ 255, 255, 255, 150 });
}

static void DrawBackdropGrid(int w, int h)
{
    for (int x = 0; x < w; x += 50) DrawLine(x, 0, x, h, (Color){ 255, 255, 255, x % 250 == 0 ? 16 : 8 });
    for (int y = 0; y < h; y += 50) DrawLine(0, y, w, y, (Color){ 255, 255, 255, y % 250 == 0 ? 16 : 8 });
}

static void DrawHUD(void)
{
    DrawText(TextFormat("Laser: %s   Beams: %d   Dispersion: %.1f   Fresnel: %s",
             colorNames[laserColor], beamOptions[beamIdx], dispersion, useFresnel ? "ON" : "OFF"),
             12, 10, 18, (Color){ 220, 225, 240, 255 });

    if (sel >= 0)
    {
        static const char *names[] = { "Slab", "Prism", "Lens", "Half-disk", "Mirror" };
        Object *o = &objs[sel];
        if (o->mirror) DrawText("Selected: Mirror", 12, 34, 18, (Color){ 255, 235, 120, 255 });
        else DrawText(TextFormat("Selected: %s   n = %.2f   scale %.2f", names[o->type], o->n, o->scale),
                      12, 34, 18, (Color){ 255, 235, 120, 255 });
    }

    if (!showHelp)
    {
        DrawText("H = help", 12, GetScreenHeight() - 26, 16, (Color){ 150, 155, 175, 255 });
        return;
    }

    const char *lines[] = {
        "CONTROLS",
        "Drag laser body = move    Drag round handle = aim (Ctrl = snap 15 deg)",
        "Mouse wheel over laser / object = rotate (Shift = fine)",
        "1 Slab   2 Prism   3 Lens   4 Half-disk   5 Mirror   (spawns at mouse)",
        "Click object = select, drag to move.  Q/E rotate   - / = scale   Del remove   K clear all",
        "Up/Down = refractive index   M = material preset (water, glass, diamond...)",
        "C = laser colour (White shows dispersion)   B = beam count   Left/Right = dispersion",
        "N = normals + angles   F = Fresnel reflections   G = grid   R = laser to mouse   H = hide",
    };
    int n = (int)(sizeof(lines) / sizeof(lines[0]));
    int px = 12, py = GetScreenHeight() - n * 20 - 22;
    DrawRectangle(px - 6, py - 8, 760, n * 20 + 16, (Color){ 0, 0, 0, 140 });
    for (int i = 0; i < n; i++)
        DrawText(lines[i], px, py + i * 20, 15, i == 0 ? (Color){ 255, 235, 120, 255 } : (Color){ 215, 220, 235, 255 });
}

// ---------------------------------------------------------------- main
typedef enum { DRAG_NONE, DRAG_EMITTER, DRAG_HANDLE, DRAG_OBJECT } DragMode;

int main(void)
{
    SetConfigFlags(FLAG_MSAA_4X_HINT | FLAG_VSYNC_HINT);
    InitWindow(1100, 700, "NewLens | LightRay Optics");
    SetTargetFPS(60);

    SpawnObject(SHAPE_PRISM, (Vector2){ 620, 350 });
    sel = -1;

    DragMode drag = DRAG_NONE;
    Vector2 dragOff = { 0, 0 };

    while (!WindowShouldClose())
    {
        float dt = GetFrameTime();
        Vector2 mouse = GetMousePosition();
        int sw = GetScreenWidth(), sh = GetScreenHeight();
        bool shift = IsKeyDown(KEY_LEFT_SHIFT) || IsKeyDown(KEY_RIGHT_SHIFT);
        bool ctrl  = IsKeyDown(KEY_LEFT_CONTROL) || IsKeyDown(KEY_RIGHT_CONTROL);

        float erad = emitter.angle * DEG2RAD;
        Vector2 edir = { cosf(erad), sinf(erad) };
        Vector2 handle = Vector2Add(emitter.pos, Vector2Scale(edir, HANDLE_DIST));
        Vector2 el = Vector2Rotate(Vector2Subtract(mouse, emitter.pos), -erad);
        bool overEmitter = fabsf(el.x) <= EMITTER_W * 0.5f && fabsf(el.y) <= EMITTER_H * 0.5f;

        // ---- mouse press
        if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT))
        {
            if (Vector2Distance(mouse, handle) < 16.0f) drag = DRAG_HANDLE;
            else if (overEmitter) { drag = DRAG_EMITTER; dragOff = Vector2Subtract(emitter.pos, mouse); }
            else
            {
                int p = PickObject(mouse);
                sel = p;
                if (p >= 0) { drag = DRAG_OBJECT; dragOff = Vector2Subtract(objs[p].pos, mouse); }
            }
        }
        if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) drag = DRAG_NONE;

        if (drag == DRAG_EMITTER)
        {
            emitter.pos = Vector2Add(mouse, dragOff);
            emitter.pos.x = Clamp(emitter.pos.x, 0, (float)sw);
            emitter.pos.y = Clamp(emitter.pos.y, 0, (float)sh);
        }
        else if (drag == DRAG_HANDLE)
        {
            float a = atan2f(mouse.y - emitter.pos.y, mouse.x - emitter.pos.x) * RAD2DEG;
            if (ctrl) a = roundf(a / 15.0f) * 15.0f;
            emitter.angle = a;
        }
        else if (drag == DRAG_OBJECT && sel >= 0)
        {
            objs[sel].pos = Vector2Add(mouse, dragOff);
        }

        // ---- wheel rotate
        float wheel = GetMouseWheelMove();
        if (wheel != 0.0f)
        {
            float step = shift ? 1.0f : 5.0f;
            if (overEmitter) emitter.angle += wheel * step;
            else
            {
                int p = PickObject(mouse);
                if (p >= 0) { sel = p; objs[p].rot += wheel * step; }
            }
        }

        // ---- keyboard
        if (IsKeyPressed(KEY_R)) emitter.pos = mouse;
        if (IsKeyPressed(KEY_ONE))   SpawnObject(SHAPE_SLAB, mouse);
        if (IsKeyPressed(KEY_TWO))   SpawnObject(SHAPE_PRISM, mouse);
        if (IsKeyPressed(KEY_THREE)) SpawnObject(SHAPE_LENS, mouse);
        if (IsKeyPressed(KEY_FOUR))  SpawnObject(SHAPE_HALFDISK, mouse);
        if (IsKeyPressed(KEY_FIVE))  SpawnObject(SHAPE_MIRROR, mouse);
        if (IsKeyPressed(KEY_C)) laserColor = (laserColor + 1) % 5;
        if (IsKeyPressed(KEY_B)) beamIdx = (beamIdx + 1) % 4;
        if (IsKeyPressed(KEY_N)) showNormals = !showNormals;
        if (IsKeyPressed(KEY_F)) useFresnel = !useFresnel;
        if (IsKeyPressed(KEY_G)) showGrid = !showGrid;
        if (IsKeyPressed(KEY_H)) showHelp = !showHelp;
        if (IsKeyPressed(KEY_K)) { objCount = 0; sel = -1; }
        if (IsKeyDown(KEY_RIGHT)) dispersion += 5.0f * dt;
        if (IsKeyDown(KEY_LEFT))  dispersion -= 5.0f * dt;
        dispersion = Clamp(dispersion, 0.0f, 25.0f);

        if (sel >= 0)
        {
            Object *o = &objs[sel];
            float rs = shift ? 10.0f : 45.0f;
            if (IsKeyDown(KEY_Q)) o->rot -= rs * dt;
            if (IsKeyDown(KEY_E)) o->rot += rs * dt;
            if (IsKeyDown(KEY_MINUS)) o->scale -= 0.6f * dt;
            if (IsKeyDown(KEY_EQUAL)) o->scale += 0.6f * dt;
            o->scale = Clamp(o->scale, 0.3f, 3.0f);
            if (!o->mirror)
            {
                if (IsKeyDown(KEY_UP))   o->n += 0.4f * dt;
                if (IsKeyDown(KEY_DOWN)) o->n -= 0.4f * dt;
                o->n = Clamp(o->n, 1.0f, 3.0f);
                if (IsKeyPressed(KEY_M))
                {
                    int next = 0;
                    for (int i = 0; i < 5; i++) if (presets[i] > o->n + 0.005f) { next = i; break; }
                    o->n = presets[next];
                }
            }
            if (IsKeyPressed(KEY_DELETE) || IsKeyPressed(KEY_BACKSPACE)) RemoveObject(sel);
        }

        for (int i = 0; i < objCount; i++) UpdateWorld(&objs[i]);
        Trace();

        // ---- draw
        BeginDrawing();
        ClearBackground((Color){ 24, 26, 38, 255 });
        if (showGrid) DrawBackdropGrid(sw, sh);

        for (int i = 0; i < objCount; i++) DrawObject(&objs[i], i == sel);
        DrawRays();
        for (int i = 0; i < objCount; i++)   // crisp outlines over the glow
            for (int k = 0; k < objs[i].count; k++)
                DrawLineEx(objs[i].world[k], objs[i].world[(k + 1) % objs[i].count], 1.5f,
                           i == sel ? (Color){ 255, 235, 120, 255 } : (Color){ 200, 230, 255, 120 });
        if (showNormals) DrawNormals();
        DrawEmitter();
        DrawHUD();
        DrawFPS(sw - 90, 10);
        EndDrawing();
    }

    CloseWindow();
    return 0;
}
