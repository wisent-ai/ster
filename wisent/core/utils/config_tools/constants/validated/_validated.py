"""
Validated constants: each is a math fact, physical definition, or protocol
spec that cannot reasonably take a different value.  Everything else lives
in arbitrary/.
"""

# --- Physical / computing facts ---
SECONDS_PER_MINUTE = 60  # definition of a minute
SECONDS_PER_HOUR = 3600  # 60 * 60
SECONDS_PER_DAY = 86400  # 60 * 60 * 24
HOURS_PER_DAY = 24  # definition of a day
MS_PER_SECOND = 1000  # SI prefix milli-
BYTES_PER_KB = 1024  # IEC binary kibibyte
BYTES_PER_MB = 1024 * 1024  # IEC binary mebibyte
BYTES_PER_GB = 1024 ** 3  # IEC binary gibibyte
HTTP_STATUS_OK = 200  # RFC 7231 section 6.3.1
HTTP_STATUS_SERVICE_UNAVAILABLE = 503  # RFC 7231 section 6.6.4

# --- Correlation / probability range boundaries ---
VIZ_CORRELATION_VMIN = -1  # Pearson r lower bound
VIZ_CORRELATION_VMAX = 1  # Pearson r upper bound
VIZ_HEATMAP_VMIN_ZERO = 0  # probability lower bound
VIZ_HEATMAP_VMAX_ONE = 1  # probability upper bound

# --- Statistical standards (textbook / convention) ---
STAT_ALPHA = 0.05  # Fisher's conventional significance level
CONFIDENCE_LEVEL = 0.95  # 1 - alpha
TARGET_POWER = 0.80  # Cohen (1988) recommended power
Z_CRITICAL_95 = 1.96  # qnorm(0.975) for two-sided 95 % CI
Z_CRITICAL_99 = 2.576  # qnorm(0.995) for two-sided 99 % CI
SIGNIFICANCE_ALPHA = 0.05  # alias used in signal detection
CI_PERCENTILE_LOW = 2.5  # lower tail of 95 % CI
CI_PERCENTILE_HIGH = 97.5  # upper tail of 95 % CI
STABILITY_Z_MARGIN = 1.96  # same z_{0.025} used for stability bands
NULL_TEST_SIGNIFICANCE_THRESHOLD = 0.05  # p < 0.05 convention
NULL_TEST_Z_SCORE_SIGNIFICANT = 2.0  # |z| >= 2 rule of thumb
Z_SCORE_SIGNIFICANCE = 2.0  # alias in part12
POWER_ADEQUATE_THRESHOLD = 0.8  # Cohen's adequate power
CHANCE_LEVEL_ACCURACY = 0.5  # binary-classification random baseline
NONSENSE_BASELINE_ACCURACY = 0.5  # same random baseline
STABILITY_BINARY_VARIANCE = 0.5  # Bernoulli variance convention

# --- Cohen's effect-size benchmarks (Cohen, 1988) ---
EFFECT_SIZE_SMALL = 0.2
EFFECT_SIZE_MEDIUM = 0.5
EFFECT_SIZE_LARGE = 0.8

# --- Metric / benchmark definitions ---
BLEU_MAX_ORDER = 4  # BLEU-4 (Papineni et al. 2002)
BLEU_MAX_N_GRAM = 4  # alias
MMLU_PRO_MAX_OPTIONS = 10  # MMLU-Pro answer choices A-J
AUDIO_SAMPLE_RATE = 16000  # 16 kHz standard for speech models
DEFAULT_AUDIO_SAMPLE_RATE = 16000  # alias
VIZ_N_COMPONENTS_2D = 2  # 2-D projection by definition
N_COMPONENTS_2D = 2  # alias
VIZ_N_COMPONENTS_3D = 3  # 3-D projection by definition
SCORE_RANGE_MIN = 0.0  # normalized score lower bound
SCORE_RANGE_MAX = 1.0  # normalized score upper bound
SCORE_SCALE_100 = 99.0  # maps [0,1] to [1,100]: score * 99 + 1

# --- Protocol / system standards ---
DEFAULT_DB_PORT = 5432  # PostgreSQL default port (IANA)
DOCKER_CPU_PERIOD_US = 100000  # Docker CFS period (100 ms)
DOCKER_TMPFS_MODE = 0o1777  # sticky + rwxrwxrwx (POSIX)
DOCKER_TMPFS_TMP_SIZE_BYTES = 134217728  # 128 MiB
DOCKER_TMPFS_WORK_SIZE_BYTES = 268435456  # 256 MiB
LLAMA_PAD_TOKEN_ID = 50256  # GPT-2 / LLaMA pad token id
SIMHASH_BIT_WIDTH = 64  # standard 64-bit SimHash
BLAKE2B_DIGEST_SIZE = 8  # 8-byte (64-bit) digest
SIMPLEQA_YEAR_DIGIT_LENGTH = 4  # yyyy format
