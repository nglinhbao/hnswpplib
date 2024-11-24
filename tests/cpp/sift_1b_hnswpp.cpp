#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "../../src/hnswppalg.h"
#include <unordered_set>

using namespace std;

// StopW class remains the same
class StopW {
    std::chrono::steady_clock::time_point time_begin;
public:
    StopW() {
        time_begin = std::chrono::steady_clock::now();
    }

    float getElapsedTimeMicro() {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count());
    }

    void reset() {
        time_begin = std::chrono::steady_clock::now();
    }
};

/*
* Author:  David Robert Nadeau
* Site:    http://NadeauSoftware.com/
* License: Creative Commons Attribution 3.0 Unported License
*          http://creativecommons.org/licenses/by/3.0/deed.en_US
*/

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))

#include <unistd.h>
#include <sys/resource.h>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
#include <fcntl.h>
#include <procfs.h>

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)

#endif

#else
#error "Cannot define getPeakRSS( ) or getCurrentRSS( ) for an unknown OS."
#endif


/**
* Returns the peak (maximum so far) resident set size (physical
* memory use) measured in bytes, or zero if the value cannot be
* determined on this OS.
*/
static size_t getPeakRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.PeakWorkingSetSize;

#elif (defined(_AIX) || defined(__TOS__AIX__)) || (defined(__sun__) || defined(__sun) || defined(sun) && (defined(__SVR4) || defined(__svr4__)))
    /* AIX and Solaris ------------------------------------------ */
    struct psinfo psinfo;
    int fd = -1;
    if ((fd = open("/proc/self/psinfo", O_RDONLY)) == -1)
        return (size_t)0L;      /* Can't open? */
    if (read(fd, &psinfo, sizeof(psinfo)) != sizeof(psinfo)) {
        close(fd);
        return (size_t)0L;      /* Can't read? */
    }
    close(fd);
    return (size_t)(psinfo.pr_rssize * 1024L);

#elif defined(__unix__) || defined(__unix) || defined(unix) || (defined(__APPLE__) && defined(__MACH__))
    /* BSD, Linux, and OSX -------------------------------------- */
    struct rusage rusage;
    getrusage(RUSAGE_SELF, &rusage);
#if defined(__APPLE__) && defined(__MACH__)
    return (size_t)rusage.ru_maxrss;
#else
    return (size_t) (rusage.ru_maxrss * 1024L);
#endif

#else
    /* Unknown OS ----------------------------------------------- */
    return (size_t)0L;          /* Unsupported. */
#endif
}


/**
* Returns the current resident set size (physical memory use) measured
* in bytes, or zero if the value cannot be determined on this OS.
*/
static size_t getCurrentRSS() {
#if defined(_WIN32)
    /* Windows -------------------------------------------------- */
    PROCESS_MEMORY_COUNTERS info;
    GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info));
    return (size_t)info.WorkingSetSize;

#elif defined(__APPLE__) && defined(__MACH__)
    /* OSX ------------------------------------------------------ */
    struct mach_task_basic_info info;
    mach_msg_type_number_t infoCount = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
        (task_info_t)&info, &infoCount) != KERN_SUCCESS)
        return (size_t)0L;      /* Can't access? */
    return (size_t)info.resident_size;

#elif defined(__linux__) || defined(__linux) || defined(linux) || defined(__gnu_linux__)
    /* Linux ---------------------------------------------------- */
    long rss = 0L;
    FILE *fp = NULL;
    if ((fp = fopen("/proc/self/statm", "r")) == NULL)
        return (size_t) 0L;      /* Can't open? */
    if (fscanf(fp, "%*s%ld", &rss) != 1) {
        fclose(fp);
        return (size_t) 0L;      /* Can't read? */
    }
    fclose(fp);
    return (size_t) rss * (size_t) sysconf(_SC_PAGESIZE);

#else
    /* AIX, BSD, Solaris, and Unknown OS ------------------------ */
    return (size_t)0L;          /* Unsupported. */
#endif
}

static void
get_gt(unsigned int *massQA,
       unsigned char *massQ,
       unsigned char *mass,
       size_t vecsize,
       size_t qsize,
       size_t vecdim,
       vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> &answers,
       size_t k) {
    
    (vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>>(qsize)).swap(answers);
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[1000 * i + j]);
        }
    }
}

static float
test_approx(unsigned char *massQ,
            size_t vecsize,
            size_t qsize,
            HNSWPP &appr_alg,
            size_t vecdim,
            vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> &answers,
            size_t k,
            float lid_threshold) {
    
    size_t correct = 0;
    size_t total = 0;

    // uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
        std::vector<float> query_vec(massQ + vecdim * i, massQ + vecdim * (i + 1));
        auto result = appr_alg.searchKnn(query_vec.data(), k, lid_threshold);
        std::priority_queue<std::pair<float, hnswlib::labeltype>> gt(answers[i]);
        unordered_set<hnswlib::labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

static void
test_vs_recall(unsigned char *massQ,
               size_t vecsize,
               size_t qsize,
               HNSWPP &appr_alg,
               size_t vecdim,
               vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> &answers,
               size_t k,
               float lid_threshold) {
    
    vector<float> lid_thresholds = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f};
    
    for (float threshold : lid_thresholds) {
        StopW stopw = StopW();
        float recall = test_approx(massQ, vecsize, qsize, appr_alg, vecdim, answers, k, threshold);
        float time_us_per_query = stopw.getElapsedTimeMicro() / qsize;

        cout << "LID threshold: " << threshold << "\tRecall: " << recall 
             << "\tTime: " << time_us_per_query << " us/query\n";
        
        if (recall > 0.99) {
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

void sift_test1B() {
    int subset_size_milllions = 200;
    int efConstruction = 40;
    int M = 16;
    int max_level = 4;
    float scale_factor = 1.0f;
    int ef_search = 20;
    float lid_threshold = 0.5f;

    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize = 10000;
    size_t vecdim = 128;
    
    char path_index[1024];
    char path_gt[1024];
    const char *path_q = "../bigann/bigann_query.bvecs";
    const char *path_data = "../bigann/bigann_base.bvecs";
    
    snprintf(path_index, sizeof(path_index), "hnswpp_sift1b_%dm_ef_%d_M_%d.bin", 
             subset_size_milllions, efConstruction, M);
    snprintf(path_gt, sizeof(path_gt), "../bigann/gnd/idx_%dM.ivecs", subset_size_milllions);

    // Load ground truth
    cout << "Loading GT:\n";
    ifstream inputGT(path_gt, ios::binary);
    unsigned int *massQA = new unsigned int[qsize * 1000];
    for (int i = 0; i < qsize; i++) {
        int t;
        inputGT.read((char *) &t, 4);
        inputGT.read((char *) (massQA + 1000 * i), t * 4);
        if (t != 1000) {
            cout << "err";
            return;
        }
    }
    inputGT.close();

    // Load queries
    cout << "Loading queries:\n";
    unsigned char *massQ = new unsigned char[qsize * vecdim];
    unsigned char *massb = new unsigned char[vecdim];
    ifstream inputQ(path_q, ios::binary);

    for (int i = 0; i < qsize; i++) {
        int in = 0;
        inputQ.read((char *) &in, 4);
        if (in != 128) {
            cout << "file error";
            exit(1);
        }
        inputQ.read((char *) massb, in);
        memcpy(massQ + i * vecdim, massb, vecdim);
    }
    inputQ.close();

    // Initialize HNSWPP
    HNSWPP *appr_alg;
    if (exists_test(path_index)) {
        cout << "Loading index from " << path_index << ":\n";
        appr_alg = new HNSWPP(vecdim, vecsize, M, efConstruction, max_level, scale_factor, ef_search);
        appr_alg->loadIndex(path_index);
        cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";
    } else {
        cout << "Building index:\n";
        appr_alg = new HNSWPP(vecdim, vecsize, M, efConstruction, max_level, scale_factor, ef_search);
        
        // Read base vectors
        std::vector<float> base_data(vecsize * vecdim);
        ifstream input(path_data, ios::binary);
        
        StopW stopw_full = StopW();
        StopW stopw = StopW();
        size_t report_every = 100000;
        
        for (size_t i = 0; i < vecsize; i++) {
            int in = 0;
            input.read((char *) &in, 4);
            if (in != 128) {
                cout << "file error";
                exit(1);
            }
            input.read((char *) massb, in);
            
            // Convert to float and copy to base_data
            for (size_t j = 0; j < vecdim; j++) {
                base_data[i * vecdim + j] = static_cast<float>(massb[j]);
            }
            
            if ((i + 1) % report_every == 0) {
                cout << (i + 1) / (0.01 * vecsize) << " %, "
                     << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                     << " Mem: " << getCurrentRSS() / 1000000 << " Mb \n";
                stopw.reset();
            }
        }
        input.close();
        
        // Prepare index and add points
        cout << "Preparing index...\n";
        appr_alg->prepareIndex(base_data.data());
        
        cout << "Adding points...\n";
        stopw.reset();
        for (size_t i = 0; i < vecsize; i++) {
            appr_alg->addPoint(base_data.data() + i * vecdim, i);
            
            if ((i + 1) % report_every == 0) {
                cout << (i + 1) / (0.01 * vecsize) << " %, "
                     << report_every / (1000.0 * 1e-6 * stopw.getElapsedTimeMicro()) << " kips "
                     << " Mem: " << getCurrentRSS() / 1000000 << " Mb \n";
                stopw.reset();
            }
        }
        
        cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << " seconds\n";
        appr_alg->saveIndex(path_index);
    }

    // Test index
    vector<std::priority_queue<std::pair<float, hnswlib::labeltype>>> answers;
    size_t k = 1;
    cout << "Parsing gt:\n";
    get_gt(massQA, massQ, nullptr, vecsize, qsize, vecdim, answers, k);
    cout << "Loaded gt\n";
    
    test_vs_recall(massQ, vecsize, qsize, *appr_alg, vecdim, answers, k, lid_threshold);
    cout << "Actual memory usage: " << getCurrentRSS() / 1000000 << " Mb \n";

    delete[] massQ;
    delete[] massb;
    delete[] massQA;
    delete appr_alg;
}

int main() {
    sift_test1B();
    return 0;
}