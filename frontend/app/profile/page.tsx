"use client";

import { useState, useEffect } from "react";
import {
  User, Save, Trash2, Plus, X, CheckCircle2, AlertTriangle,
  Loader2, Shield, Edit2, LogIn,
} from "lucide-react";
import { registerUser, getUser, updateUser, deleteUser, listAllergens } from "@/lib/api";
import type { UserResponse } from "@/lib/types";

const SRI_LANKAN_PROVINCES = [
  "Western", "Central", "Southern", "Northern", "Eastern",
  "North Western", "North Central", "Uva", "Sabaragamuwa",
];
const GENDERS = ["Male", "Female", "Other"];
const BLOOD_TYPES = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"];
const DIETARY_PATTERNS = ["Non Vegetarian", "Pescatarian", "Vegan", "Vegetarian"];
const MEDICAL_CONDITIONS = ["Asthma", "Eczema", "Food Allergy", "None of these", "Not sure"];

const ALLERGEN_GROUPS: Record<string, string[]> = {
  "Common": ["gluten", "milk_dairy", "eggs", "soy", "peanuts", "tree_nuts"],
  "Seafood": ["fish", "crustaceans", "molluscs", "shellfish"],
  "Other": ["celery", "lupin", "mustard", "sesame", "sulphites", "spices"],
};

// ─── Toast notification ───────────────────────────────────────────────────────
function Toast({ msg, type }: { msg: string; type: "success" | "error" }) {
  return (
    <div className={`fixed bottom-6 right-6 z-50 flex items-center gap-2 px-5 py-3 rounded-xl shadow-2xl fade-in text-sm font-medium ${
      type === "success"
        ? "bg-emerald-500/20 border border-emerald-500/30 text-emerald-300"
        : "bg-red-500/20 border border-red-500/30 text-red-300"
    }`}>
      {type === "success" ? <CheckCircle2 size={16} /> : <AlertTriangle size={16} />}
      {msg}
    </div>
  );
}

// ─── Profile Display Card ─────────────────────────────────────────────────────
function ProfileCard({ user, onEdit, onDelete }: {
  user: UserResponse;
  onEdit: () => void;
  onDelete: () => void;
}) {
  return (
    <div className="glass rounded-2xl overflow-hidden fade-in">
      {/* Header */}
      <div className="bg-gradient-to-r from-[rgba(255,107,53,0.15)] to-[rgba(245,158,11,0.08)] px-6 py-5 border-b border-white/5 flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#FF6B35] to-[#F59E0B] flex items-center justify-center text-white font-bold text-lg">
            {(user.full_name ?? user.email)[0].toUpperCase()}
          </div>
          <div>
            <h2 className="font-bold text-lg">{user.full_name ?? "—"}</h2>
            <p className="text-sm text-[var(--text-secondary)]">{user.email}</p>
          </div>
        </div>
        <div className="flex gap-2">
          <button onClick={onEdit} className="btn-ghost p-2 rounded-lg">
            <Edit2 size={16} />
          </button>
          <button onClick={onDelete} className="btn-ghost p-2 rounded-lg hover:border-red-500/50 hover:text-red-400">
            <Trash2 size={16} />
          </button>
        </div>
      </div>

      {/* Info grid */}
      <div className="p-6 grid grid-cols-2 gap-4">
        {[
          { label: "User ID", value: `#${user.id}` },
          { label: "Age", value: user.age ? `${user.age} yrs` : "—" },
          { label: "Gender", value: user.gender ?? "—" },
          { label: "Province", value: user.province ?? "—" },
        ].map(({ label, value }) => (
          <div key={label} className="glass rounded-xl p-4">
            <p className="text-xs text-[var(--text-muted)] mb-1">{label}</p>
            <p className="font-semibold capitalize">{value}</p>
          </div>
        ))}
      </div>

      {/* Allergens */}
      <div className="px-6 pb-6">
        <p className="text-sm text-[var(--text-muted)] mb-3 flex items-center gap-1.5">
          <Shield size={13} className="text-[var(--saffron)]" />
          Allergen Profile ({user.allergens.length})
        </p>
        <div className="flex flex-wrap gap-2">
          {user.allergens.length > 0
            ? user.allergens.map((a) => (
                <span key={a} className="chip chip-danger capitalize">{a.replace(/_/g, " ")}</span>
              ))
            : <span className="text-sm text-[var(--text-muted)]">No allergens saved</span>}
        </div>
      </div>
    </div>
  );
}

// ─── Main Profile Page ────────────────────────────────────────────────────────
export default function ProfilePage() {
  const [view, setView] = useState<"login" | "register" | "profile">("login");
  const [user, setUser] = useState<UserResponse | null>(null);
  const [allergenOptions, setAllergenOptions] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);
  const [toast, setToast] = useState<{ msg: string; type: "success" | "error" } | null>(null);

  // Form state
  const [lookupId, setLookupId] = useState("");
  const [form, setForm] = useState({
    email: "", password: "", full_name: "", age: "", gender: "", province: "",
    blood_type: "", dietary_pattern: "", outside_food_frequency: "",
  });
  const [selectedAllergens, setSelectedAllergens] = useState<string[]>([]);
  const [selectedConditions, setSelectedConditions] = useState<string[]>([]);
  const [lactoseIntolerance, setLactoseIntolerance] = useState(false);
  const [personalAllergyHistory, setPersonalAllergyHistory] = useState(false);

  useEffect(() => {
    listAllergens().then(setAllergenOptions).catch(() => {});
    const savedId = localStorage.getItem("allerscan_user_id");
    if (savedId) {
      setLookupId(savedId);
    }
  }, []);

  const showToast = (msg: string, type: "success" | "error") => {
    setToast({ msg, type });
    setTimeout(() => setToast(null), 3500);
  };

  const handleLoad = async () => {
    const id = parseInt(lookupId, 10);
    if (!id) return;
    setLoading(true);
    try {
      const u = await getUser(id);
      setUser(u);
      setView("profile");
      localStorage.setItem("allerscan_user_id", String(id));
      showToast("Profile loaded!", "success");
    } catch {
      showToast("User not found. Check your ID.", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleRegister = async () => {
    setLoading(true);
    try {
      const u = await registerUser({
        email: form.email,
        password: form.password,
        full_name: form.full_name || undefined,
        age: form.age ? parseInt(form.age, 10) : undefined,
        gender: form.gender || undefined,
        province: form.province || undefined,
        blood_type: form.blood_type || undefined,
        dietary_pattern: form.dietary_pattern || undefined,
        medical_conditions: selectedConditions,
        lactose_intolerance: lactoseIntolerance,
        outside_food_frequency: form.outside_food_frequency ? parseInt(form.outside_food_frequency, 10) : undefined,
        personal_allergy_history: personalAllergyHistory,
        allergens: selectedAllergens,
      });
      setUser(u);
      setView("profile");
      localStorage.setItem("allerscan_user_id", String(u.id));
      showToast(`Welcome, ${u.full_name ?? u.email}!`, "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Registration failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = async () => {
    if (!user) return;
    setLoading(true);
    try {
      const updated = await updateUser(user.id, {
        email: form.email || undefined,
        full_name: form.full_name || undefined,
        age: form.age ? parseInt(form.age, 10) : undefined,
        gender: form.gender || undefined,
        province: form.province || undefined,
        blood_type: form.blood_type || undefined,
        dietary_pattern: form.dietary_pattern || undefined,
        medical_conditions: selectedConditions,
        lactose_intolerance: lactoseIntolerance,
        outside_food_frequency: form.outside_food_frequency ? parseInt(form.outside_food_frequency, 10) : undefined,
        personal_allergy_history: personalAllergyHistory,
        allergens: selectedAllergens,
      });
      setUser(updated);
      setView("profile");
      showToast("Profile updated!", "success");
    } catch (err: unknown) {
      showToast(err instanceof Error ? err.message : "Update failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async () => {
    if (!user || !confirm("Delete your account permanently?")) return;
    setLoading(true);
    try {
      await deleteUser(user.id);
      setUser(null);
      setView("register");
      localStorage.removeItem("allerscan_user_id");
      showToast("Account deleted.", "success");
    } catch {
      showToast("Delete failed.", "error");
    } finally {
      setLoading(false);
    }
  };

  const startEdit = () => {
    if (!user) return;
    setForm({
      email: user.email,
      password: "",
      full_name: user.full_name ?? "",
      age: user.age ? String(user.age) : "",
      gender: user.gender ?? "",
      province: user.province ?? "",
      blood_type: user.blood_type ?? "",
      dietary_pattern: user.dietary_pattern ?? "",
      outside_food_frequency: user.outside_food_frequency !== null && user.outside_food_frequency !== undefined ? String(user.outside_food_frequency) : "",
    });
    setSelectedAllergens([...user.allergens]);
    setSelectedConditions(user.medical_conditions ? [...user.medical_conditions] : []);
    setLactoseIntolerance(user.lactose_intolerance ?? false);
    setPersonalAllergyHistory(user.personal_allergy_history ?? false);
    setView("register"); // reuse the form
  };

  const toggleAllergen = (a: string) =>
    setSelectedAllergens((prev) => prev.includes(a) ? prev.filter((x) => x !== a) : [...prev, a]);

  const toggleCondition = (c: string) =>
    setSelectedConditions((prev) => prev.includes(c) ? prev.filter((x) => x !== c) : [...prev, c]);

  const inputClass = "input-field w-full px-4 py-2.5 rounded-xl text-sm";
  const isEditing = view === "register" && !!user;

  return (
    <div className="max-w-2xl mx-auto px-6 py-12 min-h-[calc(100vh-80px)] flex flex-col justify-center">
      {toast && <Toast msg={toast.msg} type={toast.type} />}

      <div className="mb-10">
        <h1 className="text-4xl font-extrabold mb-2">
          <span className="gradient-text">My</span> Profile
        </h1>
        <p className="text-[var(--text-secondary)]">
          Save your allergen profile to personalise every scan.
        </p>
      </div>

      {/* Tab switcher */}
      {!user && (
        <div className="flex gap-2 mb-8 glass rounded-xl p-1">
          {(["login", "register"] as const).map((tab) => (
            <button
              key={tab}
              onClick={() => setView(tab)}
              className={`flex-1 py-2.5 rounded-lg text-sm font-medium transition-all capitalize ${
                view === tab
                  ? "bg-gradient-to-r from-[#FF6B35] to-[#E5531A] text-white shadow"
                  : "text-[var(--text-secondary)] hover:text-white"
              }`}
            >
              {tab === "login" ? "Load Profile" : "Create Account"}
            </button>
          ))}
        </div>
      )}

      {/* ── Profile View ── */}
      {view === "profile" && user && (
        <ProfileCard
          user={user}
          onEdit={startEdit}
          onDelete={handleDelete}
        />
      )}

      {/* ── Load Profile ── */}
      {view === "login" && (
        <div className="glass rounded-2xl p-6 space-y-4 fade-in">
          <div className="flex items-center gap-2 mb-2">
            <LogIn size={18} className="text-[var(--saffron)]" />
            <h2 className="font-semibold">Load Existing Profile</h2>
          </div>
          <div>
            <label className="block text-sm text-[var(--text-secondary)] mb-2">Your User ID</label>
            <input
              type="number"
              placeholder="e.g. 1"
              value={lookupId}
              onChange={(e) => setLookupId(e.target.value)}
              className={inputClass}
              onKeyDown={(e) => e.key === "Enter" && handleLoad()}
            />
            <p className="text-xs text-[var(--text-muted)] mt-1">
              Your ID was shown when you registered.
            </p>
          </div>
          <button
            onClick={handleLoad}
            disabled={!lookupId || loading}
            className="btn-primary w-full py-3 rounded-xl font-semibold flex items-center justify-center gap-2"
          >
            {loading ? <Loader2 size={16} className="spin" /> : <LogIn size={16} />}
            Load Profile
          </button>
          <p className="text-center text-sm text-[var(--text-muted)]">
            Don&apos;t have an account?{" "}
            <button className="text-[var(--saffron)] hover:underline" onClick={() => setView("register")}>
              Create one free
            </button>
          </p>
        </div>
      )}

      {/* ── Register / Edit Form ── */}
      {(view === "register") && (
        <div className="glass rounded-2xl p-6 space-y-5 fade-in">
          <div className="flex items-center gap-2 mb-2">
            <User size={18} className="text-[var(--saffron)]" />
            <h2 className="font-semibold">{isEditing ? "Edit Profile" : "Create Account"}</h2>
          </div>

          {/* ── Section: Account Info ── */}
          <div className="bg-white/5 border border-white/10 p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-semibold text-[var(--saffron)] uppercase tracking-wider">Account Credentials</h3>
            
            <div>
              <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Email *</label>
              <input type="email" placeholder="you@example.com" value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })} className={inputClass} />
            </div>

            {!isEditing && (
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Password * (min 6 chars)</label>
                <input type="password" placeholder="••••••" value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })} className={inputClass} />
              </div>
            )}
          </div>

          {/* ── Section: Demographics ── */}
          <div className="bg-white/5 border border-white/10 p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-semibold text-[var(--saffron)] uppercase tracking-wider">Personal Details</h3>
            
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Full Name</label>
                <input type="text" placeholder="Kamal Perera" value={form.full_name}
                  onChange={(e) => setForm({ ...form, full_name: e.target.value })} className={inputClass} />
              </div>
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Age</label>
                <input type="number" placeholder="25" value={form.age}
                  onChange={(e) => setForm({ ...form, age: e.target.value })} className={inputClass} />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Gender</label>
                <select value={form.gender} onChange={(e) => setForm({ ...form, gender: e.target.value })}
                  className={`${inputClass} appearance-none`}>
                  <option value="">Select…</option>
                  {GENDERS.map((g) => <option key={g} value={g}>{g}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Province</label>
                <select value={form.province} onChange={(e) => setForm({ ...form, province: e.target.value })}
                  className={`${inputClass} appearance-none`}>
                  <option value="">Select…</option>
                  {SRI_LANKAN_PROVINCES.map((p) => <option key={p} value={p}>{p}</option>)}
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Blood Type</label>
                <select value={form.blood_type} onChange={(e) => setForm({ ...form, blood_type: e.target.value })}
                  className={`${inputClass} appearance-none`}>
                  <option value="">Select…</option>
                  {BLOOD_TYPES.map((b) => <option key={b} value={b}>{b}</option>)}
                </select>
              </div>
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Dietary Pattern</label>
                <select value={form.dietary_pattern} onChange={(e) => setForm({ ...form, dietary_pattern: e.target.value })}
                  className={`${inputClass} appearance-none`}>
                  <option value="">Select…</option>
                  {DIETARY_PATTERNS.map((d) => <option key={d} value={d}>{d}</option>)}
                </select>
              </div>
            </div>
          </div>

          {/* ── Section: Diet & Medical ── */}
          <div className="bg-white/5 border border-white/10 p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-semibold text-[var(--saffron)] uppercase tracking-wider">Health Profile</h3>

            {/* Medical Conditions */}
            <div>
              <div className="flex items-center justify-between mb-2">
                <label className="text-sm text-[var(--text-secondary)]">Medical Conditions</label>
              </div>
              <div className="flex flex-wrap gap-2">
                {MEDICAL_CONDITIONS.map((c) => (
                  <button key={c} onClick={() => toggleCondition(c)}
                    className={`allergen-toggle px-3 py-1.5 rounded-lg text-xs font-medium capitalize ${selectedConditions.includes(c) ? "selected" : ""}`}>
                    {c}
                  </button>
                ))}
              </div>
            </div>

            {/* Extra Health Info */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-[var(--text-secondary)] mb-1.5">Outside Food Frequency (Weekly)</label>
                <select value={form.outside_food_frequency} onChange={(e) => setForm({ ...form, outside_food_frequency: e.target.value })}
                  className={`${inputClass} appearance-none`}>
                  <option value="">Select…</option>
                  {[0, 1, 2, 3, 4].map((n) => <option key={n} value={n}>{n}</option>)}
                </select>
              </div>
              <div className="flex flex-col gap-3 justify-center">
                <label className="flex items-center gap-2 cursor-pointer mt-2 text-sm text-[var(--text-secondary)] hover:text-white transition-colors">
                  <input type="checkbox" checked={lactoseIntolerance} onChange={(e) => setLactoseIntolerance(e.target.checked)} className="rounded border-white/20 bg-white/5 w-4 h-4 text-[var(--saffron)] focus:ring-[var(--saffron)] focus:ring-offset-0 transition-colors" />
                  Lactose Intolerant
                </label>
                <label className="flex items-center gap-2 cursor-pointer text-sm text-[var(--text-secondary)] hover:text-white transition-colors">
                  <input type="checkbox" checked={personalAllergyHistory} onChange={(e) => setPersonalAllergyHistory(e.target.checked)} className="rounded border-white/20 bg-white/5 w-4 h-4 text-[var(--saffron)] focus:ring-[var(--saffron)] focus:ring-offset-0 transition-colors" />
                  Personal Allergy History
                </label>
              </div>
            </div>
          </div>

          {/* ── Section: Allergens ── */}
          <div className="bg-white/5 border border-white/10 p-5 rounded-xl space-y-4">
            <h3 className="text-xs font-semibold text-[var(--saffron)] uppercase tracking-wider">Allergens</h3>
            <div className="flex items-center justify-between mb-3">
                <label className="text-sm text-[var(--text-secondary)]">Select any allergens you want to avoid</label>
                {selectedAllergens.length > 0 && (
                  <button className="text-xs bg-red-500/10 text-red-400 hover:bg-red-500/20 px-2 py-1 rounded-md transition-colors"
                    onClick={() => setSelectedAllergens([])}>Clear all</button>
                )}
              </div>
            <div className="flex flex-col gap-4">
              {Object.entries(ALLERGEN_GROUPS).map(([groupName, allergensInGroup]) => {
                const available = allergensInGroup.filter(a => allergenOptions.includes(a));
                if (available.length === 0) return null;
                return (
                  <div key={groupName}>
                    <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wider font-semibold">{groupName}</p>
                    <div className="flex flex-wrap gap-2">
                      {available.map((a) => (
                        <button key={a} onClick={() => toggleAllergen(a)}
                          className={`allergen-toggle px-3 py-1.5 rounded-lg text-xs font-medium capitalize ${selectedAllergens.includes(a) ? "selected" : ""}`}>
                          {a.replace(/_/g, " ")}
                        </button>
                      ))}
                    </div>
                  </div>
                );
              })}
              {/* Fallback for allergens not in groups */}
              {allergenOptions.filter(a => !Object.values(ALLERGEN_GROUPS).flat().includes(a)).length > 0 && (
                <div>
                    <p className="text-xs text-[var(--text-muted)] mb-2 uppercase tracking-wider font-semibold">Additional</p>
                    <div className="flex flex-wrap gap-2">
                      {allergenOptions.filter(a => !Object.values(ALLERGEN_GROUPS).flat().includes(a)).map((a) => (
                        <button key={a} onClick={() => toggleAllergen(a)}
                          className={`allergen-toggle px-3 py-1.5 rounded-lg text-xs font-medium capitalize ${selectedAllergens.includes(a) ? "selected" : ""}`}>
                          {a.replace(/_/g, " ")}
                        </button>
                      ))}
                    </div>
                </div>
              )}
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3 pt-2">
            <button
              onClick={isEditing ? handleUpdate : handleRegister}
              disabled={!form.email || (!isEditing && !form.password) || loading}
              className="btn-primary flex-1 py-3 rounded-xl font-semibold flex items-center justify-center gap-2"
            >
              {loading ? <Loader2 size={16} className="spin" /> : isEditing ? <Save size={16} /> : <Plus size={16} />}
              {isEditing ? "Save Changes" : "Create Account"}
            </button>
            {isEditing && (
              <button onClick={() => setView("profile")} className="btn-ghost px-5 py-3 rounded-xl flex items-center gap-1.5">
                <X size={15} /> Cancel
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
