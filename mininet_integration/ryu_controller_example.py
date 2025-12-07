import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mininet_integration.load_balancer import get_load_balancer

class LoadBalancedController:
    def __init__(self, algorithm='random_forest'):
        self.lb = get_load_balancer(algorithm=algorithm)
        print(f"Load balancer initialized with algorithm: {algorithm}")
    
    def get_server_for_request(self, request_info=None):
        server = self.lb.select_server()
        return server
    
    def record_response(self, server, response_time_ms, success=True):
        self.lb.update_server_stats(server, response_time_ms, success)
    
    def get_statistics(self):
        return {
            'server_loads': self.lb.get_server_loads(),
            'server_stats': dict(self.lb.server_stats)
        }

def create_controller(algorithm='random_forest'):
    return LoadBalancedController(algorithm=algorithm)

if __name__ == '__main__':
    print("Ryu Controller Load Balancer - Example Usage")
    print("=" * 60)
    print("\nThis is an example of how to integrate the load balancer")
    print("with a Ryu SDN controller.\n")
    print("Example usage in Ryu controller:")
    print("-" * 60)
    print("""
    from mininet_integration.ryu_controller_example import create_controller
    
    class MyApp(simple_switch_13):
        def __init__(self, *args, **kwargs):
            super(MyApp, self).__init__(*args, **kwargs)
            self.load_balancer = create_controller(algorithm='random_forest')
        
        def packet_in_handler(self, ev):
            # Get best server for this request
            server = self.load_balancer.get_server_for_request()
            
            # Route packet to server
            # ... routing logic ...
            
            # Record response metrics
            response_time = # ... measure response time ...
            self.load_balancer.record_response(server, response_time)
    """)

