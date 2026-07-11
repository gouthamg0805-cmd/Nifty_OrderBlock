import MetaTrader5 as mt5

# Initialize MT5 connection
if not mt5.initialize():
    print("MT5 initialization failed")
    quit()

# Get all available symbols
symbols = mt5.symbols_get()

print("Gold related symbols in XM:\n")

# Filter symbols related to gold
for symbol in symbols:
    name = symbol.name.upper()

    if "XAU" in name or "GOLD" in name:
        print(name)

# Shutdown MT5
mt5.shutdown()