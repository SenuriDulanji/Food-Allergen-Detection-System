"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { ScanLine, User, Leaf, Menu, X } from "lucide-react";
import { useState } from "react";

const NAV_LINKS = [
  { href: "/", label: "Home", icon: Leaf },
  { href: "/scan", label: "Scan Dish", icon: ScanLine },
  { href: "/profile", label: "My Profile", icon: User },
];

export default function Navbar() {
  const pathname = usePathname();
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <header className="sticky top-0 z-50 glass-strong border-b border-white/5">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link href="/" className="flex items-center gap-2 group">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-[#FF6B35] to-[#F59E0B] flex items-center justify-center shadow-lg group-hover:scale-105 transition-transform">
            <span className="text-white text-sm font-bold">A</span>
          </div>
          <span className="font-bold text-lg tracking-tight">
            <span className="gradient-text">Aller</span>
            <span className="text-[var(--text-primary)]">Scan</span>
          </span>
        </Link>

        {/* Desktop nav */}
        <nav className="hidden md:flex items-center gap-4">
          {NAV_LINKS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  active
                    ? "bg-[rgba(255,107,53,0.12)] text-[var(--saffron-light)] border border-[rgba(255,107,53,0.2)]"
                    : "text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-white/5"
                }`}
              >
                <Icon size={15} />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* CTA */}
        <div className="hidden md:flex items-center ml-4">
          <Link
            href="/scan"
            className="btn-primary px-6 py-2.5 rounded-xl text-sm font-semibold flex items-center gap-2"
          >
            <ScanLine size={16} />
            Scan Now
          </Link>
        </div>

        {/* Mobile toggle */}
        <button
          className="md:hidden p-2 rounded-lg btn-ghost"
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </div>

      {/* Mobile menu */}
      {menuOpen && (
        <div className="md:hidden glass-strong border-t border-white/5 px-6 py-4 flex flex-col gap-2">
          {NAV_LINKS.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                onClick={() => setMenuOpen(false)}
                className={`flex items-center gap-2 px-3 py-2.5 rounded-lg text-sm font-medium transition-all ${
                  active
                    ? "bg-[rgba(255,107,53,0.12)] text-[var(--saffron-light)]"
                    : "text-[var(--text-secondary)] hover:bg-white/5"
                }`}
              >
                <Icon size={16} /> {label}
              </Link>
            );
          })}
        </div>
      )}
    </header>
  );
}
