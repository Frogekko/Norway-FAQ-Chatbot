import sys
sys.path.append('/home/engebret/testPedroChat/Api')

from Api.chat__interface_group1 import create_app

Mimir_app = create_app()

if __name__ == "__main__":
    Mimir_app.run(host="0.0.0.0", port = 5000, debug=True)