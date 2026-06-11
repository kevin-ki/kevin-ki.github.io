export function LicensePill({ license }: { license: string }) {
  if (!license) return null;
  const proprietary = license.toLowerCase() === "proprietary";
  return (
    <span
      className="inline-block whitespace-nowrap rounded-full border px-2 py-0.5 text-[10px] font-medium leading-4 tracking-wide"
      style={
        proprietary
          ? {
              color: "#E8716D",
              borderColor: "rgba(232, 113, 109, 0.35)",
              background: "rgba(232, 113, 109, 0.08)",
            }
          : {
              color: "#33B386",
              borderColor: "rgba(51, 179, 134, 0.35)",
              background: "rgba(51, 179, 134, 0.08)",
            }
      }
    >
      {license}
    </span>
  );
}
