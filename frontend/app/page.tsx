import Link from "next/link";
import { ScanLine, Brain, Shield, Zap, ChevronRight, Star } from "lucide-react";

const FEATURES = [
  {
    icon: ScanLine,
    color: "from-[#FF6B35] to-[#F59E0B]",
    title: "AI Vision Analysis",
    desc: "Gemini 3.1 Flash Lite identifies Sri Lankan dishes from a single photo with remarkable accuracy.",
  },
  {
    icon: Brain,
    color: "from-[#8B5CF6] to-[#60A5FA]",
    title: "ML Risk Prediction",
    desc: " logistic regression models trained on real survey data predict your personalised clinical allergy risk.",
  },
  {
    icon: Shield,
    color: "from-[#10B981] to-[#34D399]",
    title: "Apriori Safety Net",
    desc: "Validated association rules flag cross-reactive allergens — the invisible risks most systems miss.",
  },
  {
    icon: Zap,
    color: "from-[#F59E0B] to-[#FCD34D]",
    title: "RAG Knowledge Base",
    desc: "ChromaDB retrieval gives the AI deep dish-specific context for every scan — zero hallucinations.",
  },
];

const STATS = [
  { label: "Allergens Tracked", value: "24" },
  { label: "Sri Lankan Dishes", value: "35" },
  { label: "ML Models", value: "24" },
  { label: "Detection Layers", value: "4" },
];

export default function HomePage() {
  return (
    <div className="min-h-screen flex items-center justify-center py-20">
    <div className="max-w-6xl w-full mx-auto px-6 flex flex-col gap-10">
      
      {/* ── Hero ───────────────────────────────────────────────────────── */}
      <section className="text-center relative pt-10">
        {/* Background glow orb */}
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="w-[600px] h-[600px] rounded-full bg-[var(--saffron)] opacity-[0.04] blur-[120px]" />
        </div>

        <div className="relative">
          <div className="inline-flex items-center gap-2 mb-8 px-5 py-2.5 rounded-full glass border border-[rgba(255,107,53,0.2)] text-sm text-[var(--saffron-light)]">
            <span className="pulse-dot green" />
            <span>4-Layer AI Detection System · Powered by Gemini 3.1 Flash Lite</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight leading-[1.1] mb-8">
            <span className="text-[var(--text-primary)]">Know What's </span>
            <span className="gradient-text">In Your Plate.</span>
          </h1>

          <p className="text-lg md:text-xl text-[var(--text-secondary)]">
            Snap a photo of any Sri Lankan dish. Our AI instantly detects allergens,
            predicts your personal clinical risk, and warns you about hidden cross-reactivities —
            so you can eat confidently.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-10 my-16">
          <Link
            href="/scan"
            className="btn-primary px-12 py-6 rounded-2xl text-lg font-semibold flex items-center gap-2 justify-center transition-transform hover:scale-105"
          >
            <ScanLine size={25} />
            Start Scanning Free
            <ChevronRight size={18} />
          </Link>
          
          <Link
            href="/profile"
            className="btn-ghost px-12 py-6 rounded-2xl text-lg font-medium flex items-center gap-2 w-full sm:w-auto justify-center transition-transform hover:scale-105"
          >
            Create Profile
          </Link>
        </div>

        </div>
      </section>

      {/* ── Stats ──────────────────────────────────────────────────────── */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-6">
        {STATS.map(({ label, value }) => (
          <div key={label} className="glass rounded-2xl p-8 text-center glow-saffron">
            <div className="text-4xl font-extrabold gradient-text mb-2">{value}</div>
            <div className="text-sm md:text-base text-[var(--text-secondary)]">{label}</div>
          </div>
        ))}
      </section>

      {/* ── Features ───────────────────────────────────────────────────── */}
      <section>
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold mb-4">
            Four Layers of Protection
          </h2>
          <p className="text-lg text-[var(--text-secondary)]">
            A hybrid pipeline that combines vision AI, machine learning, and association mining
            — all working together to keep you safe.
          </p>
        </div>

        <div className="grid md:grid-cols-2 gap-8">
          {FEATURES.map(({ icon: Icon, color, title, desc }) => (
            <div
              key={title}
              className="glass rounded-3xl p-8 flex gap-6 hover:border-[rgba(255,107,53,0.2)] transition-all group"
            >
              <div className={`w-14 h-14 rounded-2xl bg-gradient-to-br ${color} flex items-center justify-center shrink-0 shadow-lg group-hover:scale-110 transition-transform`}>
                <Icon size={26} className="text-white" />
              </div>
              <div>
                <h3 className="text-xl font-semibold text-[var(--text-primary)] mb-2">{title}</h3>
                <p className="text-base text-[var(--text-secondary)] leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ── CTA Banner ─────────────────────────────────────────────────── */}
      <section>
        <div className="glass rounded-[2.5rem] p-12 md:p-16 text-center relative overflow-hidden glow-saffron">
          <div className="absolute inset-0 bg-gradient-to-br from-[rgba(255,107,53,0.08)] to-transparent pointer-events-none" />
          <div className="relative">
            <div className="flex justify-center gap-1 mb-6">
              {[...Array(5)].map((_, i) => (
                <Star key={i} size={24} className="text-[var(--gold)] fill-[var(--gold)]" />
              ))}
            </div>
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Ready to scan your first dish?
            </h2>
            <p className="text-lg text-[var(--text-secondary)] mb-10 max-w-2xl mx-auto leading-relaxed">
              Create a profile to save your allergen preferences, or jump straight in and
              scan any Sri Lankan dish photo right now.
            </p>
            <Link
              href="/scan"
              className="btn-primary px-12 py-5 rounded-2xl text-lg font-semibold inline-flex items-center gap-3 transition-transform hover:scale-105"
            >
              <ScanLine size={20} /> Scan a Dish Now
            </Link>
          </div>
        </div>
      </section>
      
    </div>
</div>
  );
}
