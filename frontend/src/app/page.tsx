import { Inter } from "next/font/google";

const inter = Inter({ subsets: ["latin"] });

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white shadow-md">
        <div className="container mx-auto px-4 py-4 flex justify-between items-center">
          <h1 className="text-xl md:text-2xl font-bold">
            Supply Chain Optimization
          </h1>
          <nav className="hidden md:block">
            <ul className="flex space-x-6">
              <li>
                <a href="#" className="hover:text-blue-200">
                  Dashboard
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-200">
                  Inventory
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-200">
                  Forecasting
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-200">
                  Analytics
                </a>
              </li>
              <li>
                <a href="#" className="hover:text-blue-200">
                  Settings
                </a>
              </li>
            </ul>
          </nav>
          <button className="md:hidden">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
              className="w-6 h-6"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M4 6h16M4 12h16M4 18h16"
              />
            </svg>
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-8">
        {/* Dashboard Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-gray-500 text-sm font-medium mb-2">
              Total Inventory
            </h3>
            <p className="text-3xl font-bold text-gray-800">34,218</p>
            <p className="text-sm text-green-600 mt-2">↑ 12% from last month</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-gray-500 text-sm font-medium mb-2">
              Pending Orders
            </h3>
            <p className="text-3xl font-bold text-gray-800">423</p>
            <p className="text-sm text-red-600 mt-2">↑ 4% from last month</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-gray-500 text-sm font-medium mb-2">
              On-Time Delivery
            </h3>
            <p className="text-3xl font-bold text-gray-800">94.2%</p>
            <p className="text-sm text-green-600 mt-2">↑ 2% from last month</p>
          </div>

          <div className="bg-white p-6 rounded-lg shadow-md">
            <h3 className="text-gray-500 text-sm font-medium mb-2">
              Forecast Accuracy
            </h3>
            <p className="text-3xl font-bold text-gray-800">89%</p>
            <p className="text-sm text-yellow-600 mt-2">← No change</p>
          </div>
        </div>

        {/* Main Dashboard Sections */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Supply Chain Map */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-2">
            <h2 className="text-xl font-semibold mb-4">Supply Chain Network</h2>
            <div className="bg-gray-100 h-80 rounded-md flex items-center justify-center">
              <p className="text-gray-500">Interactive Supply Chain Map</p>
            </div>
          </div>

          {/* Recent Alerts */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Recent Alerts</h2>
            <ul className="space-y-4">
              <li className="pb-4 border-b">
                <div className="flex items-center">
                  <div className="h-3 w-3 rounded-full bg-red-500 mr-3"></div>
                  <p className="text-sm font-medium">Supplier delay detected</p>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Raw materials shipment delayed by 3 days
                </p>
              </li>
              <li className="pb-4 border-b">
                <div className="flex items-center">
                  <div className="h-3 w-3 rounded-full bg-yellow-500 mr-3"></div>
                  <p className="text-sm font-medium">
                    Inventory threshold alert
                  </p>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Component X-1092 below safety stock
                </p>
              </li>
              <li className="pb-4 border-b">
                <div className="flex items-center">
                  <div className="h-3 w-3 rounded-full bg-blue-500 mr-3"></div>
                  <p className="text-sm font-medium">New order received</p>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Order #39281 from Client A
                </p>
              </li>
              <li>
                <div className="flex items-center">
                  <div className="h-3 w-3 rounded-full bg-green-500 mr-3"></div>
                  <p className="text-sm font-medium">Optimization complete</p>
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Route optimization saved 12% in transport costs
                </p>
              </li>
            </ul>
          </div>

          {/* Forecasting Chart */}
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Demand Forecasting</h2>
            <div className="bg-gray-100 h-64 rounded-md flex items-center justify-center">
              <p className="text-gray-500">Demand Forecast Chart</p>
            </div>
          </div>

          {/* Inventory Status */}
          <div className="bg-white p-6 rounded-lg shadow-md lg:col-span-2">
            <h2 className="text-xl font-semibold mb-4">Inventory Status</h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Product
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      SKU
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Location
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Quantity
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Component A
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      SKU-12345
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      Warehouse 1
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      5,240
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        Optimal
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Component B
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      SKU-23456
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      Warehouse 2
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      3,890
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-yellow-100 text-yellow-800">
                        Low
                      </span>
                    </td>
                  </tr>
                  <tr>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                      Component C
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      SKU-34567
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      Warehouse 1
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      7,120
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                      <span className="px-2 inline-flex text-xs leading-5 font-semibold rounded-full bg-green-100 text-green-800">
                        Optimal
                      </span>
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="bg-gray-100 border-t mt-12">
        <div className="container mx-auto px-4 py-6">
          <p className="text-center text-gray-500 text-sm">
            © 2024 Supply Chain Optimization. All rights reserved.
          </p>
        </div>
      </footer>
    </div>
  );
}
